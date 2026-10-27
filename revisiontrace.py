#!/usr/bin/env python3
"""
RevisionTrace - does a correction reach every conclusion that depended on it,
while unrelated conclusions stay untouched?

Pure Python, standard library only. No network, no APIs, no third-party packages.

Pieces
------
* Graph      : facts -> conclusions (AND / OR / k-of-n rules) -> decisions (sinks)
* Evidence   : a SET of claims (add / retract with tombstones), so the final
               evidence is independent of the order updates arrive in
* Oracle     : computes the expected state straight from the final evidence set
* Strategies : full_recompute, incremental (both correct by design) and four
               history-dependent updaters that are CONTROLLED FAILURE MODELS
* Scorer     : exact agreement, stale, wrong-direction, collateral,
               false resolution, order stability, update cost

Usage
-----
    python revisiontrace.py                       # 6 categories x 50 cases, seed 2026
    python revisiontrace.py --seed 7 --cases 100
    python revisiontrace.py --selftest            # unit checks only
"""
import argparse
import csv
import hashlib
import heapq
import itertools
import json
import math
import os
import random
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from xml.sax.saxutils import escape

TRUE, FALSE, UNKNOWN = "TRUE", "FALSE", "UNKNOWN"
CATEGORIES = [
    "single_fact_change",
    "multi_support",
    "shared_dependency",
    "conflicting_evidence",
    "unrelated_change",
    "multi_update",
]


# --------------------------------------------------------------------------- #
# Graph
# --------------------------------------------------------------------------- #
class Graph:
    def __init__(self):
        self.kind, self.inputs, self.rule = {}, {}, {}
        self.depth, self.comp, self.idx = {}, {}, {}
        self.order = []
        self.dependents = defaultdict(list)
        self._nf = self._nc = 0

    def _register(self, n, kind, inputs, rule, depth, comp):
        self.kind[n], self.inputs[n], self.rule[n] = kind, list(inputs), rule
        self.depth[n], self.comp[n] = depth, comp
        self.idx[n] = len(self.order)
        self.order.append(n)

    def add_fact(self, comp):
        n = f"F{self._nf}"
        self._nf += 1
        self._register(n, "fact", [], None, 0, comp)
        return n

    def add_concl(self, inputs, rule, comp):
        n = f"C{self._nc}"
        self._nc += 1
        depth = 1 + max(self.depth[i] for i in inputs)
        self._register(n, "concl", inputs, rule, depth, comp)
        for i in inputs:
            self.dependents[i].append(n)
        return n

    def attach(self, concl, fact):
        self.inputs[concl].append(fact)
        self.dependents[fact].append(concl)

    def facts(self):
        return [n for n in self.order if self.kind[n] == "fact"]

    def concls(self):
        return [n for n in self.order if self.kind[n] == "concl"]

    def sinks(self):
        return [n for n in self.concls() if not self.dependents[n]]

    def descendants(self, n):
        seen, stack = set(), [n]
        while stack:
            x = stack.pop()
            for d in self.dependents[x]:
                if d not in seen:
                    seen.add(d)
                    stack.append(d)
        return seen

    def ancestors(self, n):
        seen, stack = set(), [n]
        while stack:
            x = stack.pop()
            for i in self.inputs[x]:
                if i not in seen:
                    seen.add(i)
                    stack.append(i)
        return seen

    def describe(self, n):
        r = self.rule[n]
        k = k_of(r, len(self.inputs[n]))
        name = "AND" if r[0] == "AND" else "OR" if r[0] == "OR" else f"{k}-of"
        return f"{name}({','.join(self.inputs[n])})"

    def to_json(self):
        return {
            n: {
                "kind": self.kind[n],
                "comp": self.comp[n],
                "depth": self.depth[n],
                "inputs": self.inputs[n],
                "rule": list(self.rule[n]) if self.rule[n] else None,
            }
            for n in self.order
        }


def k_of(rule, n):
    return n if rule[0] == "AND" else 1 if rule[0] == "OR" else rule[1]


# --------------------------------------------------------------------------- #
# Evidence: a set of claims with tombstones (order-independent by construction)
#
#   ("ADD", claim_id, fact, TRUE|FALSE, rank)
#   ("RETRACT", claim_id)
#
# Fact value rule: among live claims on a fact take the highest rank; if all
# claims at that rank agree the fact has that value; if they disagree the fact
# is UNKNOWN (unresolved conflict); no live claims -> UNKNOWN.
# A retraction is permanent for that claim id, even if it arrives before the ADD.
# --------------------------------------------------------------------------- #
class Evidence:
    def __init__(self):
        self.claims = {}
        self.by_fact = {}
        self.tomb = set()

    def copy(self):
        e = Evidence()
        e.claims = dict(self.claims)
        e.by_fact = {k: set(v) for k, v in self.by_fact.items()}
        e.tomb = set(self.tomb)
        return e

    def apply(self, u):
        """Apply an update; return the fact whose claim set changed, else None."""
        if u[0] == "ADD":
            _, cid, fact, val, rank = u
            if cid in self.tomb or cid in self.claims:
                return None
            self.claims[cid] = (fact, val, rank)
            self.by_fact.setdefault(fact, set()).add(cid)
            return fact
        _, cid = u
        self.tomb.add(cid)
        c = self.claims.pop(cid, None)
        if c is None:
            return None
        self.by_fact[c[0]].discard(cid)
        return c[0]

    def live(self, fact):
        return [(cid,) + self.claims[cid] for cid in sorted(self.by_fact.get(fact, ()))]

    def live_all(self):
        return [(cid,) + self.claims[cid]
                for cid in sorted(self.claims, key=lambda c: int(c[1:]))]

    def fact_value(self, fact):
        cl = [self.claims[c] for c in self.by_fact.get(fact, ())]
        if not cl:
            return UNKNOWN
        top = max(c[2] for c in cl)
        vals = {c[1] for c in cl if c[2] == top}
        return next(iter(vals)) if len(vals) == 1 else UNKNOWN

    def to_json(self):
        return [{"id": c, "fact": v[0], "value": v[1], "rank": v[2]}
                for c, v in sorted(self.claims.items())]


# --------------------------------------------------------------------------- #
# Oracle: expected state from the final evidence set (independent code path:
# explicit Kleene operators and a "k-th best value" formulation of thresholds)
# --------------------------------------------------------------------------- #
_ORD = {TRUE: 2, UNKNOWN: 1, FALSE: 0}


def kleene_and(vs):
    if FALSE in vs:
        return FALSE
    return TRUE if all(v == TRUE for v in vs) else UNKNOWN


def kleene_or(vs):
    if TRUE in vs:
        return TRUE
    return FALSE if all(v == FALSE for v in vs) else UNKNOWN


def oracle_rule(rule, vs):
    if rule[0] == "AND":
        return kleene_and(vs)
    if rule[0] == "OR":
        return kleene_or(vs)
    kth = sorted(vs, key=lambda v: -_ORD[v])[rule[1] - 1]  # k-th best input
    return kth


def oracle_state(g, ev):
    memo = {}

    def val(n):
        if n not in memo:
            if g.kind[n] == "fact":
                memo[n] = ev.fact_value(n)
            else:
                memo[n] = oracle_rule(g.rule[n], [val(i) for i in g.inputs[n]])
        return memo[n]

    return {n: val(n) for n in g.order}


# --------------------------------------------------------------------------- #
# Revision strategies. Each starts from the CORRECT stored state and then
# processes updates one at a time. cost = number of node evaluations.
# --------------------------------------------------------------------------- #
class Strategy:
    name = "base"
    exact_by_design = False

    def __init__(self, g, ev, state):
        self.g, self.ev, self.state, self.cost = g, ev, dict(state), 0

    def eval_node(self, n):
        self.cost += 1
        if self.g.kind[n] == "fact":
            return self.ev.fact_value(n)
        vs = [self.state[i] for i in self.g.inputs[n]]
        k = k_of(self.g.rule[n], len(vs))
        t, u = vs.count(TRUE), vs.count(UNKNOWN)
        return TRUE if t >= k else FALSE if t + u < k else UNKNOWN

    def apply(self, update):
        f = self.ev.apply(update)
        if f is not None:
            self.on_fact(f)

    def answer(self):
        """Answering from scratch: derive every conclusion from the evidence
        currently held, with no prior conclusions. The same derivation rules
        every strategy uses; it is the 'we taught AI to answer' half."""
        for n in self.g.order:
            self.state[n] = self.eval_node(n)

    def on_fact(self, f):
        raise NotImplementedError


class FullRecompute(Strategy):
    name, exact_by_design = "full_recompute", True

    def on_fact(self, f):
        for n in self.g.order:
            self.state[n] = self.eval_node(n)


class Incremental(Strategy):
    name, exact_by_design = "incremental", True

    def children(self, n):
        return self.g.dependents[n]

    def adjust(self, n, new):
        return new

    def on_fact(self, f):
        new = self.eval_node(f)
        if new == self.state[f]:
            return
        self.state[f] = new
        heap, seen = [], set()

        def push(n):
            for c in self.children(n):
                if c not in seen:
                    seen.add(c)
                    heapq.heappush(heap, (self.g.idx[c], c))

        push(f)
        while heap:
            _, n = heapq.heappop(heap)
            new = self.adjust(n, self.eval_node(n))
            if new != self.state[n]:
                self.state[n] = new
                push(n)


class HistoryShallow(Strategy):
    """Revises only direct dependents of the changed fact (one hop, no cascade)."""
    name = "history_shallow"

    def on_fact(self, f):
        new = self.eval_node(f)
        if new == self.state[f]:
            return
        self.state[f] = new
        for d in self.g.dependents[f]:
            self.state[d] = self.eval_node(d)


class HistoryBranchFirst(Incremental):
    """Cascades correctly, but follows only the first dependent of each node."""
    name = "history_branch_first"
    exact_by_design = False

    def children(self, n):
        return self.g.dependents[n][:1]


class HistorySticky(Incremental):
    """Cascades correctly, but never retreats from a decided belief to UNKNOWN."""
    name = "history_sticky"
    exact_by_design = False

    def adjust(self, n, new):
        if new == UNKNOWN and self.state[n] != UNKNOWN:
            return self.state[n]
        return new


class HistoryFlush(Strategy):
    """On a change: wipe every conclusion to UNKNOWN, then refill only the
    descendants of the changed fact (blunt invalidation + narrow recompute)."""
    name = "history_flush"

    def on_fact(self, f):
        new = self.eval_node(f)
        if new == self.state[f]:
            return
        self.state[f] = new
        for c in self.g.concls():
            self.state[c] = UNKNOWN
        desc = self.g.descendants(f)
        for n in self.g.order:
            if n in desc:
                self.state[n] = self.eval_node(n)


STRATEGIES = [FullRecompute, Incremental, HistoryShallow,
              HistoryBranchFirst, HistorySticky, HistoryFlush]


class HopLimited(Incremental):
    """Ablation: propagate a correction at most `limit` hops from the changed fact.
    limit=0 updates the evidence and never touches a stored conclusion (an
    append-only memory that answers but never revises)."""
    limit = 0
    exact_by_design = False

    def on_fact(self, f):
        new = self.eval_node(f)
        if new == self.state[f]:
            return
        self.state[f] = new
        hop, heap, seen = {f: 0}, [], set()

        def push(n):
            if hop[n] >= self.limit:
                return
            for c in self.g.dependents[n]:
                hop[c] = min(hop.get(c, 99), hop[n] + 1)
                if c not in seen:
                    seen.add(c)
                    heapq.heappush(heap, (self.g.idx[c], c))

        push(f)
        while heap:
            _, n = heapq.heappop(heap)
            new = self.eval_node(n)
            if new != self.state[n]:
                self.state[n] = new
                push(n)


def hop_class(k):
    return type(f"hops_{k}", (HopLimited,), {"limit": k, "name": f"hops_{k}"})


# --------------------------------------------------------------------------- #
# Case generation (seeded, deterministic)
# --------------------------------------------------------------------------- #
def pick_rule(rng, n):
    r = rng.random()
    if n >= 3 and r < 0.4:
        return ("K", rng.randint(2, n - 1))
    return ("AND",) if r < 0.7 else ("OR",)


def add_component(g, rng, comp, n_facts, layers, width):
    levels = [[g.add_fact(comp) for _ in range(n_facts)]]
    for _ in range(layers):
        cur, prev = [], levels[-1]
        pool = [n for lv in levels for n in lv]
        for _ in range(rng.randint(max(1, width - 1), width)):
            n_in = rng.randint(2, min(4, len(pool)))
            ins = [rng.choice(prev)]
            others = [n for n in pool if n not in ins]
            ins += rng.sample(others, min(n_in - 1, len(others)))
            cur.append(g.add_concl(ins, pick_rule(rng, len(ins)), comp))
        levels.append(cur)
    for f in levels[0]:  # every fact must matter to at least one conclusion
        if not g.dependents[f]:
            g.attach(rng.choice(levels[1]), f)


def build_graph(rng, idx):
    g = Graph()
    add_component(g, rng, 0, rng.randint(5, 8), 2 + idx % 4, rng.randint(2, 4))
    add_component(g, rng, 1, rng.randint(3, 4), 2, 2)  # independent padding branch
    return g


def init_evidence(g, rng, ids):
    ev = Evidence()
    for f in g.facts():
        if rng.random() < 0.9:
            ev.apply(("ADD", f"c{next(ids)}", f, TRUE if rng.random() < 0.6 else FALSE, 1))
    return ev


def style_updates(rng, ev, fact, ids):
    """Updates that change the evidence for one fact (retract / replace / override)."""
    live = ev.live(fact)
    if not live:
        return [("ADD", f"c{next(ids)}", fact, rng.choice([TRUE, FALSE]), 1)]
    top = max(l[3] for l in live)
    topc = [l for l in live if l[3] == top]
    opp = FALSE if topc[0][2] == TRUE else TRUE
    style = rng.choice(["retract", "replace", "override"])
    if style == "retract":
        return [("RETRACT", l[0]) for l in topc]
    if style == "replace":
        return [("RETRACT", l[0]) for l in topc] + [("ADD", f"c{next(ids)}", fact, opp, top)]
    return [("ADD", f"c{next(ids)}", fact, opp, top + 1)]


def eval_updates(g, ev0, ups):
    s0 = oracle_state(g, ev0)
    ev = ev0.copy()
    for u in ups:
        ev.apply(u)
    s1 = oracle_state(g, ev)
    return s0, s1, {c for c in g.concls() if s0[c] != s1[c]}


def main_facts(g):
    return [n for n in g.facts() if g.comp[n] == 0]


def gen_single(rng, g, ev0, ids):
    f = rng.choice(main_facts(g))
    ups = style_updates(rng, ev0, f, ids)
    _, _, ch = eval_updates(g, ev0, ups)
    return ups if ch and max(g.depth[c] for c in ch) >= 2 else None


def gen_multi_support(rng, g, ev0, ids):
    f = rng.choice(main_facts(g))
    ups = style_updates(rng, ev0, f, ids)
    s0, s1, ch = eval_updates(g, ev0, ups)
    if s0[f] == s1[f]:
        return None
    survivors = [c for c in g.dependents[f] if len(g.inputs[c]) >= 2 and c not in ch]
    return ups if survivors else None


def gen_shared(rng, g, ev0, ids):
    cands = [f for f in main_facts(g) if len(g.dependents[f]) >= 2]
    if not cands:
        return None
    f = rng.choice(cands)
    ups = style_updates(rng, ev0, f, ids)
    _, _, ch = eval_updates(g, ev0, ups)
    branches = sum(1 for d in g.dependents[f] if ({d} | g.descendants(d)) & ch)
    return ups if branches >= 2 else None


def gen_conflict(rng, g, ev0, ids):
    cands = [f for f in main_facts(g) if ev0.live(f)]
    f = rng.choice(cands)
    live = ev0.live(f)
    top = max(l[3] for l in live)
    cur = [l for l in live if l[3] == top][0]
    opp = FALSE if cur[2] == TRUE else TRUE
    ups = [("ADD", f"c{next(ids)}", f, opp, top)]  # equal-rank contradiction
    if rng.random() < 0.5:  # resolved later by higher-ranked evidence
        ups.append(("ADD", f"c{next(ids)}", f, rng.choice([TRUE, FALSE]), top + 1))
    _, _, ch = eval_updates(g, ev0, ups)
    return ups if ch else None


def gen_unrelated(rng, g, ev0, ids):
    f = rng.choice([n for n in g.facts() if g.comp[n] == 1])
    ups = style_updates(rng, ev0, f, ids)
    _, _, ch = eval_updates(g, ev0, ups)
    return ups if ch else None


def gen_multi_update(rng, g, ev0, ids):
    pool = rng.sample(main_facts(g), min(3, len(main_facts(g))))
    ev, ups = ev0.copy(), []
    target = rng.randint(4, 6)
    while len(ups) < target:
        step = style_updates(rng, ev, rng.choice(pool), ids)
        for u in step:
            ev.apply(u)
        ups.extend(step)
    if rng.random() < 0.25:  # transient claim added then retracted
        cid = f"c{next(ids)}"
        pos = rng.randint(0, len(ups))
        ups.insert(pos, ("ADD", cid, rng.choice(main_facts(g)), rng.choice([TRUE, FALSE]), 9))
        ups.insert(rng.randint(pos + 1, len(ups)), ("RETRACT", cid))
    s0, s1, ch = eval_updates(g, ev0, ups)
    if not ch:
        return None
    ev = ev0.copy()
    for u in ups[:-1]:
        ev.apply(u)
        mid = oracle_state(g, ev)
        if any(mid[c] != s1[c] for c in g.concls()):
            return ups
    return None


GEN = {
    "single_fact_change": gen_single,
    "multi_support": gen_multi_support,
    "shared_dependency": gen_shared,
    "conflicting_evidence": gen_conflict,
    "unrelated_change": gen_unrelated,
    "multi_update": gen_multi_update,
}


class Case:
    pass


def make_case(seed, category, idx):
    for attempt in range(5000):
        rng = random.Random(f"{seed}|{category}|{idx}|{attempt}")
        g = build_graph(rng, idx)
        ids = itertools.count()
        ev0 = init_evidence(g, rng, ids)
        ups = GEN[category](rng, g, ev0, ids)
        if ups is None:
            continue
        c = Case()
        c.id, c.category, c.g, c.ev0, c.updates = f"{category}-{idx:03d}", category, g, ev0, ups
        c.s0, c.s1, c.changed = eval_updates(g, ev0, ups)
        c.depth = max(g.depth.values())
        return c
    raise RuntimeError(f"could not generate {category} #{idx}")


def case_to_json(c):
    return {
        "id": c.id, "category": c.category, "graph": c.g.to_json(),
        "initial_evidence": c.ev0.to_json(),
        "updates": [list(u) for u in c.updates],
        "expected_initial": {n: c.s0[n] for n in c.g.concls()},
        "expected_final": {n: c.s1[n] for n in c.g.concls()},
        "must_change": sorted(c.changed),
        "must_stay": sorted(set(c.g.concls()) - c.changed),
    }


# --------------------------------------------------------------------------- #
# Running and scoring
# --------------------------------------------------------------------------- #
def run_strategy(cls, case, order):
    ev = case.ev0.copy()
    s = cls(case.g, ev, oracle_state(case.g, ev))
    for i in order:
        s.apply(case.updates[i])
    return {n: s.state[n] for n in case.g.concls()}, s.cost


def final_evidence(case):
    ev = case.ev0.copy()
    for u in case.updates:
        ev.apply(u)
    return ev


def answer_from_scratch(cls, case):
    """Same system, same FINAL evidence, but no history: it just answers."""
    s = cls(case.g, final_evidence(case), {n: UNKNOWN for n in case.g.order})
    s.answer()
    return {n: s.state[n] for n in case.g.concls()}


def score(case, final):
    g, s0, s1 = case.g, case.s0, case.s1
    r = dict(should_change=0, should_stay=0, stale=0, wrong_dir=0, collateral=0,
             expected_unknown=0, false_res=0, exact=True)
    for n in g.concls():
        i, e, s = s0[n], s1[n], final[n]
        r["should_change" if e != i else "should_stay"] += 1
        if e == UNKNOWN:
            r["expected_unknown"] += 1
            if s != UNKNOWN:
                r["false_res"] += 1
        if s != e:
            r["exact"] = False
            if e == i:
                r["collateral"] += 1
            elif s == i:
                r["stale"] += 1
            else:
                r["wrong_dir"] += 1
    sinks_ok = all(final[n] == s1[n] for n in g.sinks())
    r["hidden"] = (not r["exact"]) and sinks_ok  # decisions right, interior wrong
    return r


def orders_for(case, n_perm):
    n = len(case.updates)
    base = list(range(n))
    if n < 2:
        return [base]
    rng = random.Random(f"perm|{case.id}")
    seen, out = {tuple(base)}, [base]
    for _ in range(n_perm * 4):
        if len(out) > n_perm:
            break
        p = base[:]
        rng.shuffle(p)
        if tuple(p) not in seen:
            seen.add(tuple(p))
            out.append(p)
    return out


FIELDS = ["cases", "exact", "answer_exact", "should_change", "stale", "should_stay", "collateral",
          "wrong_dir", "expected_unknown", "false_res", "failures", "hidden",
          "order_cases", "order_stable", "order_correct", "cost", "updates"]


def new_agg():
    return {k: 0 for k in FIELDS}


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p, d = k / n, 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d


def run_all(cases, n_perm):
    agg, per_case, results = {}, [], {}
    for case in cases:
        for cls in STRATEGIES:
            orders = orders_for(case, n_perm)
            finals, costs = [], []
            for o in orders:
                f, c = run_strategy(cls, case, o)
                finals.append(f)
                costs.append(c)
            final, cost = finals[0], costs[0]
            sc = score(case, final)
            ans = answer_from_scratch(cls, case)
            ans_ok = ans == {n: case.s1[n] for n in case.g.concls()}
            sc["answer_exact"] = ans_ok
            results[(case.id, cls.name)] = (final, sc)
            stable = all(f == finals[0] for f in finals)
            correct = all(f == {n: case.s1[n] for n in case.g.concls()} for f in finals)
            per_case.append({
                "case_id": case.id, "category": case.category, "strategy": cls.name,
                "n_conclusions": len(case.g.concls()), "max_depth": case.depth,
                "n_updates": len(case.updates), "exact": int(sc["exact"]),
                "answer_exact": int(ans_ok),
                "stale": sc["stale"], "wrong_dir": sc["wrong_dir"],
                "collateral": sc["collateral"], "false_res": sc["false_res"],
                "order_stable": int(stable) if len(orders) > 1 else "",
                "order_correct": int(correct) if len(orders) > 1 else "",
                "cost": cost})
            for key in [(cls.name, case.category), (cls.name, "ALL"),
                        (cls.name, f"depth={case.depth}")]:
                a = agg.setdefault(key, new_agg())
                a["cases"] += 1
                a["exact"] += sc["exact"]
                a["answer_exact"] += ans_ok
                a["failures"] += not sc["exact"]
                a["hidden"] += sc["hidden"]
                for k in ("should_change", "stale", "should_stay", "collateral",
                          "wrong_dir", "expected_unknown", "false_res"):
                    a[k] += sc[k]
                a["cost"] += cost
                a["updates"] += len(case.updates)
                if len(orders) > 1:
                    a["order_cases"] += 1
                    a["order_stable"] += stable
                    a["order_correct"] += correct
    return agg, per_case, results


def pct(a, b):
    return 100.0 * a / b if b else float("nan")


def summary_rows(agg):
    rows = []
    for (strat, cat), a in sorted(agg.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        lo, hi = wilson(a["exact"], a["cases"])
        rows.append({
            "strategy": strat, "category": cat, "cases": a["cases"], "exact": a["exact"],
            "exact_pct": round(pct(a["exact"], a["cases"]), 1),
            "answer_from_scratch_pct": round(pct(a["answer_exact"], a["cases"]), 1),
            "revision_gap_pts": round(pct(a["answer_exact"], a["cases"])
                                      - pct(a["exact"], a["cases"]), 1),
            "exact_ci_lo": round(100 * lo, 1), "exact_ci_hi": round(100 * hi, 1),
            "stale": a["stale"], "should_change": a["should_change"],
            "stale_rate_pct": round(pct(a["stale"], a["should_change"]), 1),
            "collateral": a["collateral"], "should_stay": a["should_stay"],
            "collateral_rate_pct": round(pct(a["collateral"], a["should_stay"]), 1),
            "wrong_direction": a["wrong_dir"],
            "false_resolution": a["false_res"], "expected_unknown": a["expected_unknown"],
            "hidden_damage": a["hidden"], "failures": a["failures"],
            "order_cases": a["order_cases"],
            "order_stable_pct": round(pct(a["order_stable"], a["order_cases"]), 1),
            "order_correct_pct": round(pct(a["order_correct"], a["order_cases"]), 1),
            "mean_cost_per_update": round(a["cost"] / a["updates"], 2) if a["updates"] else 0,
        })
    return rows


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def fmt(x, w=7, d=1):
    return f"{x:{w}.{d}f}" if x == x else " " * (w - 3) + "n/a"


def print_report(agg, cases, digest):
    names = [c.name for c in STRATEGIES]
    cats = CATEGORIES + ["ALL"]
    print(f"\nRevisionTrace  |  {len(cases)} cases  |  case-set hash {digest}\n")
    print("TABLE 0  same system, same FINAL evidence, two routes (all cases pooled)")
    print(f"{'strategy':22}{'answer from':>14}{'revise from':>14}{'gap':>8}")
    print(f"{'':22}{'scratch %':>14}{'old state %':>14}{'(pts)':>8}")
    for s in names:
        a = agg[(s, 'ALL')]
        ans, rev = pct(a['answer_exact'], a['cases']), pct(a['exact'], a['cases'])
        print(f"{s:22}{ans:14.1f}{rev:14.1f}{ans - rev:8.1f}")
    print("  Answering is a function of the evidence. Revising is a function of the history.\n")
    print("TABLE 1  exact final-state agreement, % of cases (update order as generated)")
    print(f"{'strategy':22}" + "".join(f"{c[:13]:>14}" for c in cats))
    for s in names:
        print(f"{s:22}" + "".join(
            f"{pct(agg[(s, c)]['exact'], agg[(s, c)]['cases']):14.1f}" for c in cats))
    print("\nTABLE 2  all cases pooled")
    h = ["exact%", "95% CI", "stale%", "collat%", "wrongdir", "falseres", "hidden%",
         "ord-stab%", "ord-corr%", "cost/upd"]
    print(f"{'strategy':22}" + "".join(f"{x:>10}" for x in h))
    for s in names:
        a = agg[(s, "ALL")]
        lo, hi = wilson(a["exact"], a["cases"])
        row = [fmt(pct(a["exact"], a["cases"])), f"{100*lo:.0f}-{100*hi:.0f}",
               fmt(pct(a["stale"], a["should_change"])),
               fmt(pct(a["collateral"], a["should_stay"])),
               str(a["wrong_dir"]), str(a["false_res"]),
               fmt(pct(a["hidden"], a["failures"])),
               fmt(pct(a["order_stable"], a["order_cases"])),
               fmt(pct(a["order_correct"], a["order_cases"])),
               f"{a['cost']/a['updates']:.1f}"]
        print(f"{s:22}" + "".join(f"{x:>10}" for x in row))
    print("  stale% = stale / conclusions that should change; collat% = collateral / conclusions"
          "\n  that should stay; hidden% = failures where every decision (sink) was still correct.")
    depths = sorted({k[1] for k in agg if k[1].startswith("depth=")})
    print("\nTABLE 3  exact agreement % by graph depth")
    print(f"{'strategy':22}" + "".join(f"{d:>10}" for d in depths))
    for s in names:
        print(f"{s:22}" + "".join(
            f"{pct(agg[(s, d)]['exact'], agg[(s, d)]['cases']):10.1f}" for d in depths))


def traces(cases, results):
    out = []
    by_id = {c.id: c for c in cases}
    for cls in STRATEGIES:
        for cat in CATEGORIES:
            fails = [c for c in cases if c.category == cat
                     and not results[(c.id, cls.name)][1]["exact"]]
            if not fails:
                continue
            c = min(fails, key=lambda x: len(x.g.concls()))
            final, sc = results[(c.id, cls.name)]
            g = c.g
            out.append(f"=== {cls.name} | {cat} | {c.id} ===")
            out.append("Updates (evidence is a set; arrival order must not matter):")
            for i, u in enumerate(c.updates, 1):
                if u[0] == "ADD":
                    out.append(f"  u{i}: ADD {u[1]}: {u[2]} := {u[3]} (rank {u[4]})")
                else:
                    out.append(f"  u{i}: RETRACT {u[1]}")
            ev1 = c.ev0.copy()
            for u in c.updates:
                ev1.apply(u)
            moves = [f"{f} {c.s0[f]}->{c.s1[f]}" for f in g.facts() if c.s0[f] != c.s1[f]]
            out.append("Fact changes: " + (", ".join(moves) if moves else "none"))
            out.append("Same final evidence, answered from scratch by this same system: "
                       + ("all conclusions correct" if sc["answer_exact"] else "NOT all correct"))
            out.append("Conclusions that changed or were wrong (initial -> expected | system):")
            shown = 0
            for n in g.concls():
                i, e, s = c.s0[n], c.s1[n], final[n]
                if i == e and s == e:
                    continue
                tag = "ok" if s == e else (
                    "COLLATERAL" if e == i else "STALE" if s == i else "WRONG-DIRECTION")
                out.append(f"  {n:4} {g.describe(n):30} {i:>7} -> {e:<7} | {s:<7} {tag}")
                shown += 1
            out.append(f"  ({len(g.concls()) - shown} conclusions unchanged and correct, omitted)")
            if sc["hidden"]:
                out.append("  NOTE: every final decision (sink) is correct; the damage is interior.")
            out.append("")
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# Ablation: what teaches a system to revise?
# Same cases, same incremental machinery, but a correction may only travel
# k hops from the changed fact. Exactness should need k >= graph depth.
# --------------------------------------------------------------------------- #
ABLATION_K = [0, 1, 2, 3, 4, 5]


def hop_label(name):
    return {"hops_0": "hops_0 (never revises)",
            "incremental": "incremental (unbounded)"}.get(name, name)


def run_ablation(cases):
    classes = [hop_class(k) for k in ABLATION_K] + [Incremental]
    table = {}
    for cls in classes:
        for case in cases:
            final, cost = run_strategy(cls, case, list(range(len(case.updates))))
            ok = score(case, final)["exact"]
            for key in ("ALL", f"depth={case.depth}"):
                a = table.setdefault((cls.name, key),
                                     {"cases": 0, "exact": 0, "cost": 0, "updates": 0})
                a["cases"] += 1
                a["exact"] += ok
                a["cost"] += cost
                a["updates"] += len(case.updates)
    return [c.name for c in classes], table


def print_ablation(names, table):
    depths = sorted({k[1] for k in table if k[1].startswith("depth=")})
    print("\nTABLE 4  what teaches it to revise? exact agreement % by how far a correction may travel")
    print(f"{'propagation limit':26}" + "".join(f"{d:>10}" for d in depths)
          + f"{'ALL':>10}{'cost/upd':>10}")
    for n in names:
        row = "".join(f"{pct(table[(n, d)]['exact'], table[(n, d)]['cases']):10.1f}" for d in depths)
        a = table[(n, "ALL")]
        print(f"{hop_label(n):26}{row}{pct(a['exact'], a['cases']):10.1f}"
              f"{a['cost'] / a['updates']:10.1f}")
    print("  A correction must travel to the end of the longest dependency chain it touches.")


def ablation_rows(names, table):
    rows = []
    for n in names:
        for key in sorted(k[1] for k in table if k[0] == n):
            a = table[(n, key)]
            rows.append({"propagation": n, "group": key, "cases": a["cases"],
                         "exact": a["exact"], "exact_pct": round(pct(a["exact"], a["cases"]), 1),
                         "mean_cost_per_update": round(a["cost"] / a["updates"], 2)})
    return rows


# --------------------------------------------------------------------------- #
# SVG figures: what changed, what should have changed, what the system held
# --------------------------------------------------------------------------- #
FILL = {"fact_changed": "#a8c7fa", "ok": "#a8dab5", "stale": "#f6aea9",
        "collateral": "#fde293", "wrong": "#d7aefb", "same": "#f1f3f4"}
TAG = {"ok": "ok", "same": "ok", "stale": "STALE", "collateral": "COLLATERAL", "wrong": "WRONG DIRECTION"}


def trace_nodes(case, final):
    g = case.g
    core = {f for f in g.facts() if case.s0[f] != case.s1[f]}       # what the evidence changed
    wrong = {n for n in g.concls() if final[n] != case.s1[n]}       # what the system got wrong
    core |= wrong
    for n in wrong:                                                  # the chain that should have carried it
        core |= {a for a in g.ancestors(n) if case.s0[a] != case.s1[a]}
    nodes = set(core)
    for n in core:
        nodes |= set(g.inputs[n])
    return nodes


def node_status(case, final, n):
    if case.g.kind[n] == "fact":
        return "fact_changed" if case.s0[n] != case.s1[n] else "same"
    i, e, s = case.s0[n], case.s1[n], final[n]
    if s == e:
        return "ok" if e != i else "same"
    return "collateral" if e == i else "stale" if s == i else "wrong"


def pick_trace(cases, results, strategy, want):
    """Smallest failing case showing a stale (with a correctly revised ancestor) or collateral node."""
    best = None
    for c in cases:
        final, sc = results[(c.id, strategy)]
        if sc["exact"]:
            continue
        g = c.g
        if want == "stale":
            ok = False
            for n in g.concls():
                if c.s0[n] != c.s1[n] and final[n] == c.s0[n]:
                    if any(g.kind[a] == "concl" and c.s0[a] != c.s1[a] and final[a] == c.s1[a]
                           for a in g.ancestors(n)):
                        ok = True
            if not ok:
                continue
        elif not any(c.s0[n] == c.s1[n] and final[n] != c.s1[n] for n in g.concls()):
            continue
        nodes = trace_nodes(c, final)
        clean = all(c.s0[n] != UNKNOWN and c.s1[n] != UNKNOWN for n in nodes)
        key = (not clean, len(nodes), len(c.updates))
        if best is None or key < best[0]:
            best = (key, c, final)
    return (best[1], best[2]) if best else None


def render_svg(case, final, title):
    g = case.g
    nodes = trace_nodes(case, final)
    bw, bh, gx, gy = 250, 62, 80, 22
    cols = sorted({g.depth[n] for n in nodes})
    rows = {d: [n for n in sorted(nodes, key=lambda x: g.idx[x]) if g.depth[n] == d] for d in cols}
    pos = {}
    for ci, d in enumerate(cols):
        for ri, n in enumerate(rows[d]):
            pos[n] = (30 + ci * (bw + gx), 64 + ri * (bh + gy))
    width = max(760, 30 + len(cols) * (bw + gx) - gx + 30)
    height = 64 + max(len(r) for r in rows.values()) * (bh + gy) + 84
    p = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
         f'width="{width}" height="{height}" font-family="Menlo, Consolas, monospace">',
         f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
         f'<text x="30" y="28" font-size="15" font-weight="bold" fill="#202124">{escape(title)}</text>',
         '<text x="30" y="46" font-size="11" fill="#5f6368">each box: value before -> value the '
         'oracle expects, then what the system holds after the updates</text>']
    for n in sorted(nodes, key=lambda x: g.idx[x]):
        if g.kind[n] == "concl":
            for i in g.inputs[n]:
                if i in pos:
                    x1, y1 = pos[i][0] + bw, pos[i][1] + bh / 2
                    x2, y2 = pos[n][0], pos[n][1] + bh / 2
                    p.append(f'<path d="M{x1} {y1} C{x1 + gx / 2} {y1} {x2 - gx / 2} {y2} {x2} {y2}" '
                             'fill="none" stroke="#9aa0a6" stroke-width="1.4"/>')
    for n, (x, y) in pos.items():
        st = node_status(case, final, n)
        if g.kind[n] == "fact":
            l1 = f"{n}  (evidence)"
            l2 = (f"{case.s0[n]} -> {case.s1[n]}" if case.s0[n] != case.s1[n]
                  else f"{case.s0[n]} (unchanged)")
            l3 = ""
        else:
            l1 = f"{n}  {g.describe(n)}"
            l1 = l1 if len(l1) <= 34 else l1[:33] + "…"
            l2 = f"expected {case.s0[n]} -> {case.s1[n]}"
            l3 = f"system holds {final[n]}  [{TAG[st]}]"
        p.append(f'<rect x="{x}" y="{y}" width="{bw}" height="{bh}" rx="6" fill="{FILL[st]}" '
                 'stroke="#5f6368" stroke-width="1"/>')
        for k, line in enumerate((l1, l2, l3)):
            if line:
                weight = ' font-weight="bold"' if k == 0 else ""
                p.append(f'<text x="{x + 10}" y="{y + 20 + 18 * k}" font-size="11" '
                         f'fill="#202124"{weight}>{escape(line)}</text>')
    legend = [("fact_changed", "changed evidence"), ("ok", "revised correctly"),
              ("stale", "STALE"), ("collateral", "COLLATERAL"),
              ("wrong", "WRONG DIRECTION"), ("same", "unchanged, correct")]
    ly = height - 62
    for k, (key, label) in enumerate(legend):
        lx, yy = 30 + (k % 3) * ((width - 60) // 3), ly + (k // 3) * 24
        p.append(f'<rect x="{lx}" y="{yy}" width="16" height="16" rx="3" fill="{FILL[key]}" stroke="#5f6368"/>')
        p.append(f'<text x="{lx + 24}" y="{yy + 12}" font-size="11" fill="#202124">{escape(label)}</text>')
    p.append("</svg>")
    return "\n".join(p)


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #
def selftest():
    T_, F_, U_ = TRUE, FALSE, UNKNOWN
    k2 = ("K", 2)
    assert oracle_rule(k2, [T_, U_, F_]) == U_
    assert oracle_rule(k2, [T_, T_, F_]) == T_
    assert oracle_rule(k2, [T_, F_, F_]) == F_
    assert oracle_rule(k2, [U_, U_, F_]) == U_
    assert oracle_rule(("AND",), [T_, F_, U_]) == F_
    assert oracle_rule(("AND",), [T_, T_, U_]) == U_
    assert oracle_rule(("OR",), [F_, U_]) == U_
    assert oracle_rule(("OR",), [F_, F_]) == F_
    # engine threshold form must agree with the oracle form on every input
    rng = random.Random(1)
    for _ in range(5000):
        n = rng.randint(2, 5)
        vs = [rng.choice([T_, F_, U_]) for _ in range(n)]
        rule = rng.choice([("AND",), ("OR",)] + ([("K", rng.randint(2, n - 1))] if n >= 3 else []))
        k, t, u = k_of(rule, n), vs.count(T_), vs.count(U_)
        eng = T_ if t >= k else F_ if t + u < k else U_
        assert eng == oracle_rule(rule, vs), (rule, vs)
    # evidence semantics
    ev = Evidence()
    ev.apply(("ADD", "a", "F0", T_, 1))
    ev.apply(("ADD", "b", "F0", F_, 1))
    assert ev.fact_value("F0") == U_                      # equal-rank conflict
    ev.apply(("ADD", "c", "F0", T_, 2))
    assert ev.fact_value("F0") == T_                      # higher rank wins
    ev.apply(("RETRACT", "c"))
    assert ev.fact_value("F0") == U_                      # older conflict resurfaces
    e1, e2 = Evidence(), Evidence()
    for u in [("ADD", "x", "F1", T_, 1), ("RETRACT", "x")]:
        e1.apply(u)
    for u in [("RETRACT", "x"), ("ADD", "x", "F1", T_, 1)]:
        e2.apply(u)
    assert e1.fact_value("F1") == e2.fact_value("F1") == U_  # tombstones commute
    # generation is deterministic
    a = hashlib.sha256(json.dumps(case_to_json(make_case(5, "multi_update", 3)),
                                  sort_keys=True).encode()).hexdigest()
    b = hashlib.sha256(json.dumps(case_to_json(make_case(5, "multi_update", 3)),
                                  sort_keys=True).encode()).hexdigest()
    assert a == b
    print("selftest: ok")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--cases", type=int, default=50, help="cases per category")
    ap.add_argument("--perms", type=int, default=5, help="extra update orderings per case")
    ap.add_argument("--out", default="revisiontrace_out")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    selftest()
    if args.selftest:
        return

    cases = [make_case(args.seed, cat, i) for cat in CATEGORIES for i in range(args.cases)]
    blob = json.dumps([case_to_json(c) for c in cases], sort_keys=True)
    digest = hashlib.sha256(blob.encode()).hexdigest()[:12]

    agg, per_case, results = run_all(cases, args.perms)

    # the two correct-by-design strategies must match the independent oracle everywhere
    for c in cases:
        for cls in STRATEGIES:
            if cls.exact_by_design:
                assert results[(c.id, cls.name)][1]["exact"], (cls.name, c.id)
    print(f"verified: full_recompute and incremental match the oracle on all {len(cases)} cases")
    for c in cases:
        for cls in STRATEGIES:
            assert results[(c.id, cls.name)][1]["answer_exact"], (cls.name, c.id)
    print(f"verified: all {len(STRATEGIES)} strategies answer correctly from scratch on all {len(cases)} cases")

    print_report(agg, cases, digest)

    abl_names, abl = run_ablation(cases)
    assert abl[("hops_5", "ALL")]["exact"] == len(cases)       # deep enough reach is exact
    assert abl[("incremental", "ALL")]["exact"] == len(cases)
    # never revising is right exactly when nothing needed to change (survival-only cases)
    assert abl[("hops_0", "ALL")]["exact"] == sum(1 for c in cases if not c.changed)
    print_ablation(abl_names, abl)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "ablation.csv"), "w", newline="") as f:
        arows = ablation_rows(abl_names, abl)
        w = csv.DictWriter(f, fieldnames=list(arows[0]))
        w.writeheader()
        w.writerows(arows)
    figures = []
    for strat, want, fname in [("history_shallow", "stale", "featured_stale.svg"),
                               ("history_flush", "collateral", "featured_collateral.svg")]:
        pick = pick_trace(cases, results, strat, want)
        if pick:
            c, final = pick
            svg = render_svg(c, final, f"{strat}  |  {c.id}  |  {len(c.updates)} update(s)")
            ET.fromstring(svg)  # must be well-formed XML
            with open(os.path.join(args.out, fname), "w", encoding="utf-8") as f:
                f.write(svg)
            figures.append(f"{fname} ({c.id})")
    rows = summary_rows(agg)
    with open(os.path.join(args.out, "results_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(args.out, "per_case.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_case[0]))
        w.writeheader()
        w.writerows(per_case)
    with open(os.path.join(args.out, "cases.json"), "w") as f:
        json.dump({"seed": args.seed, "hash": digest,
                   "cases": [case_to_json(c) for c in cases]}, f, indent=1)
    with open(os.path.join(args.out, "failure_traces.txt"), "w") as f:
        f.write(traces(cases, results))
    print(f"\nwrote results_summary.csv, per_case.csv, ablation.csv, cases.json, "
          f"failure_traces.txt, {', '.join(figures)} to {args.out}/")


if __name__ == "__main__":
    sys.exit(main())
