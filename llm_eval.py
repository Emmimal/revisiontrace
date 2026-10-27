#!/usr/bin/env python3
"""
RevisionTrace, language-model arm (OPTIONAL). Standard library only; this script
makes no network calls and uses no API.

It puts a language model through three routes to the same final evidence:

  answer       the model sees only the FINAL evidence and derives every conclusion
  revise       the model sees the EARLIER evidence, its stored conclusions, and the
               evidence updates, and must report the conclusions that now hold
  revise_hint  the same as revise, plus a dependency record: which facts the updates
               touched and which conclusions depend on them, in the order to re-derive
               them. The harness does the bookkeeping; the model only does the derivation.

All prompts use the same rules and end at the same final evidence, and every reply is
scored against the same oracle. This is a separate experiment: nothing here is mixed
with the controlled-strategy results from revisiontrace.py.

How to read the three rows
  answer low           the model cannot derive the conclusions, so revision cannot be isolated
  revise ~ copy-stored it is mostly copying its stored conclusions (the baseline row shows this)
  revise_hint > revise a dependency record helps: the missing piece is bookkeeping

Commands
--------
  demo    prove the pipeline works using dummy "models" (no model needed)
  export  write prompts.jsonl, to paste into any model yourself
  run     feed each prompt to a LOCAL program on stdin and collect its stdout
            python llm_eval.py run --cmd "ollama run llama3" --label llama3
          Refuses to overwrite an existing responses file. Use --append to add
          conditions or resume an interrupted run, or --overwrite to start again.
            python llm_eval.py run --cmd "..." --conditions revise_hint --append
  score   score a responses.jsonl file

responses.jsonl: one JSON object per line: {"id": ..., "condition": "answer"|"revise"|
"revise_hint", "response": "<model text>"}. The model must reply with one line per
conclusion, e.g. "C3: TRUE". Report the model, version, and settings you used with the
results. Use the same --seed, --per-category and --max-conclusions for export, run and score.
"""
import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from collections import defaultdict

import revisiontrace as rt

CONDS = ("answer", "revise", "revise_hint")

RULES = """You maintain conclusions derived from evidence. Every fact and conclusion is TRUE, FALSE or UNKNOWN.

Rules
- A fact's value comes from its claims. Among the claims on a fact, only those with the highest rank count. If they all agree, the fact has that value. If they disagree, the fact is UNKNOWN. A fact with no claims is UNKNOWN.
- AND(...): FALSE if any input is FALSE; TRUE if all inputs are TRUE; otherwise UNKNOWN.
- OR(...): TRUE if any input is TRUE; FALSE if all inputs are FALSE; otherwise UNKNOWN.
- K-of(...): TRUE if at least K inputs are TRUE; FALSE if, even counting every UNKNOWN input as TRUE, fewer than K inputs could be TRUE; otherwise UNKNOWN.
- Conclusions are computed from the values of their inputs, which may be facts or other conclusions."""

FORMAT = ("Reply with exactly one line per conclusion, in the order listed, in the form "
          "`C1: TRUE`. Output nothing else.")


def rule_text(g, n):
    r = g.rule[n]
    ins = ", ".join(g.inputs[n])
    if r[0] == "K":
        return f"{n} = {r[1]}-of({ins})"
    return f"{n} = {r[0]}({ins})"


def claim_text(cid, fact, val, rank):
    return f"{cid}: {fact} is {val} (rank {rank})"


def update_text(u):
    if u[0] == "ADD":
        return "ADD " + claim_text(u[1], u[2], u[3], u[4])
    return f"RETRACT {u[1]}"


def dependency_record(case):
    """Facts the updates touched, and the conclusions that depend on them (directly or
    through other conclusions), in an order that re-derives inputs before dependents.
    It says nothing about what the new values are."""
    ev = case.ev0.copy()
    touched = []
    for u in case.updates:
        f = ev.apply(u)
        if f is not None and f not in touched:
            touched.append(f)
    touched.sort(key=lambda n: case.g.idx[n])
    affected = set()
    for f in touched:
        affected |= case.g.descendants(f)
    return touched, [n for n in case.g.concls() if n in affected]


def hint_text(case):
    touched, affected = dependency_record(case)
    if not touched:
        return ("Dependency record from the system: the updates touched no facts, so no "
                "conclusion depends on them. Every stored value is unchanged.")
    lines = ["Dependency record from the system:",
             f"- The updates touched these facts: {', '.join(touched)}.",
             "- Conclusions that depend on them, directly or through other conclusions, "
             "in the order to re-derive them: " + (", ".join(affected) if affected else "none") + ".",
             "- Every conclusion not in that list does not depend on a touched fact, so its "
             "stored value is unchanged.",
             "Re-derive the listed conclusions in the order given, using the current evidence. "
             "Then report every conclusion, copying the stored value for each one that is not "
             "listed."]
    return "\n".join(lines)


def build_prompt(case, cond):
    g = case.g
    out = [RULES, "", "Conclusions and how they are derived:"]
    out += [f"- {rule_text(g, n)}" for n in g.concls()]
    if cond == "answer":
        ev = rt.final_evidence(case)
        out += ["", "Evidence (all current claims):"]
        out += [f"- {claim_text(*c)}" for c in ev.live_all()] or ["- (none)"]
        out += ["", "Derive the value of every conclusion from this evidence."]
    else:
        out += ["", "Earlier evidence:"]
        out += [f"- {claim_text(*c)}" for c in case.ev0.live_all()] or ["- (none)"]
        out += ["", "Conclusions you stored from the earlier evidence:"]
        out += [f"- {n}: {case.s0[n]}" for n in g.concls()]
        out += ["", "New evidence updates:"]
        out += [f"{i}. {update_text(u)}" for i, u in enumerate(case.updates, 1)]
        out += ["", "The current evidence is the earlier evidence plus every added claim, minus every "
                    "retracted claim. A retracted claim is gone for good, and the order of the updates "
                    "does not matter."]
        if cond == "revise_hint":
            out += ["", hint_text(case)]
        out += ["", "Report the value of every conclusion after these updates."]
    out += ["", FORMAT]
    return "\n".join(out)


LINE = re.compile(r"^[\s>*`\-]*\**(C\d+)\**\s*[:=]\s*\**\s*(TRUE|FALSE|UNKNOWN)\b", re.I)
ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def parse_response(text, names):
    got = {}
    for line in ANSI.sub("", text).splitlines():
        m = LINE.match(line)
        if m and m.group(1).upper() in names:
            got[m.group(1).upper()] = m.group(2).upper()   # last mention wins
    return got


def build_cases(seed, per_cat, max_conclusions=None):
    cases = [rt.make_case(seed, cat, i) for cat in rt.CATEGORIES for i in range(per_cat)]
    if max_conclusions:   # deterministic filter: keep only small graphs
        cases = [c for c in cases if len(c.g.concls()) <= max_conclusions]
    return cases


# --------------------------------------------------------------------------- #
def evaluate(cases, responses):
    stats = {c: defaultdict(int) for c in CONDS}
    ok, cat_ok = {}, defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for case in cases:
        names = case.g.concls()
        for cond in CONDS:
            text = responses.get((case.id, cond))
            if text is None:
                continue
            parsed = parse_response(text, set(names))
            final = {n: parsed.get(n, "INVALID") for n in names}
            sc = rt.score(case, final)
            missing = [n for n in names if n not in parsed]
            sc["false_res"] -= sum(1 for n in missing if case.s1[n] == rt.UNKNOWN)
            s = stats[cond]
            s["cases"] += 1
            s["exact"] += sc["exact"]
            s["nodes"] += len(names)
            s["nodes_ok"] += sum(1 for n in names if final[n] == case.s1[n])
            s["unparsed_nodes"] += len(missing)
            for k in ("should_change", "stale", "should_stay", "collateral",
                      "wrong_dir", "false_res", "expected_unknown"):
                s[k] += sc[k]
            ok[(case.id, cond)] = sc["exact"]
            cell = cat_ok[cond][case.category]
            cell[0] += sc["exact"]
            cell[1] += 1
    return stats, ok, cat_ok


def row(name, s, answer_route=False):
    lo, hi = rt.wilson(s["exact"], s["cases"])
    if answer_route:   # the model never sees the old state, so stale/collateral do not apply
        st = co = f"{'n/a':>9}"
    else:
        st = f"{rt.pct(s['stale'], s['should_change']):9.1f}"
        co = f"{rt.pct(s['collateral'], s['should_stay']):9.1f}"
    return (f"{name:18}{s['cases']:6d}{rt.pct(s['exact'], s['cases']):9.1f}"
            f"{f'{100 * lo:.0f}-{100 * hi:.0f}':>9}"
            f"{rt.pct(s['nodes_ok'], s['nodes']):11.1f}{st}{co}"
            f"{s['wrong_dir']:10d}{s['false_res']:10d}{s['unparsed_nodes']:10d}")


def paired(cases, ok, a, b, label_a, label_b):
    both = [c for c in cases if (c.id, a) in ok and (c.id, b) in ok]
    if not both:
        return
    rr = sum(1 for c in both if ok[(c.id, a)] and ok[(c.id, b)])
    ra = sum(1 for c in both if ok[(c.id, a)] and not ok[(c.id, b)])
    rb = sum(1 for c in both if not ok[(c.id, a)] and ok[(c.id, b)])
    ww = sum(1 for c in both if not ok[(c.id, a)] and not ok[(c.id, b)])
    print(f"  {label_a} vs {label_b}, {len(both)} cases: both right {rr} | only {label_a} right {ra} "
          f"| only {label_b} right {rb} | both wrong {ww}")
    if rr + ra:
        print(f"    of the {rr + ra} cases right on {label_a}, wrong on {label_b}: {ra} "
              f"({100 * ra / (rr + ra):.1f}%)"
              + ("   (fewer than 15 cases: do not quote)" if rr + ra < 15 else ""))


def carry_unlisted(cases, responses):
    """For the hint condition only: the prompt says conclusions it does not list keep their
    stored value, so a missing line is read as 'unchanged' instead of as an error. Models that
    list only the affected conclusions are otherwise scored as if they had damaged the rest."""
    out = dict(responses)
    for c in cases:
        t = responses.get((c.id, "revise_hint"))
        if t is None:
            continue
        names = c.g.concls()
        parsed = parse_response(t, set(names))
        out[(c.id, "revise_hint")] = "\n".join(f"{n}: {parsed.get(n, c.s0[n])}" for n in names)
    return out


def report(cases, responses, label):
    stats, ok, cat_ok = evaluate(cases, responses)
    print(f"\n=== {label} ===")
    print(f"{'condition':18}{'cases':>6}{'exact%':>9}{'95% CI':>9}{'node acc%':>11}{'stale%':>9}"
          f"{'collat%':>9}{'wrongdir':>10}{'falseres':>10}{'unparsed':>10}")
    for cond in CONDS:
        if stats[cond]["cases"]:
            print(row(cond, stats[cond], answer_route=(cond == "answer")))
    carried = False
    if stats["revise_hint"]["cases"] and stats["revise_hint"]["unparsed_nodes"]:
        st2, _, _ = evaluate(cases, carry_unlisted(cases, responses))
        print(row("revise_hint+carry", st2["revise_hint"]))
        carried = True
    if any(stats[c]["cases"] for c in CONDS if c != "answer"):
        base, _, _ = evaluate(cases, dummy_responses(cases, "stale"))
        if base["revise"]["cases"]:
            print(row("copy-stored baseline", base["revise"]))
    print("  exact% = every conclusion in the case right; node acc% = share of individual "
          "conclusions right.\n  stale% and collat% apply to the revise routes only. The baseline "
          "row is a system that\n  copies its stored conclusions and never revises, on the same cases.")
    if carried:
        print("  revise_hint+carry: the hint says unlisted conclusions keep their stored value, so a "
              "missing line is\n  read as 'unchanged'. The plain revise_hint row scores a missing line "
              "as an error. Report both.")
    print("\npaired comparisons (read only if there are enough cases)")
    paired(cases, ok, "answer", "revise", "answer", "revise")
    paired(cases, ok, "revise", "revise_hint", "revise", "hint")
    present = [c for c in CONDS if stats[c]["cases"]]
    print("\nexact% by category (" + " / ".join(present) + ")")
    for cat in rt.CATEGORIES:
        cells = []
        for cond in present:
            x, n = cat_ok[cond][cat]
            cells.append(f"{100 * x / n:5.1f}" if n else "  n/a")
        print(f"  {cat:22}" + " / ".join(cells))


# --------------------------------------------------------------------------- #
def dummy_responses(cases, kind):
    """perfect: derives and revises, with or without the hint.
       stale: answers perfectly from scratch, never revises, ignores the hint.
       assisted: answers perfectly, never revises unaided, revises perfectly with the hint."""
    out = {}
    for c in cases:
        names = c.g.concls()
        oracle = "\n".join(f"{n}: {c.s1[n]}" for n in names)
        stored = "\n".join(f"{n}: {c.s0[n]}" for n in names)
        out[(c.id, "answer")] = oracle
        out[(c.id, "revise")] = oracle if kind == "perfect" else stored
        out[(c.id, "revise_hint")] = oracle if kind in ("perfect", "assisted") else stored
    return out


def read_responses(path):
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                out[(r["id"], r["condition"])] = r["response"]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["demo", "export", "run", "score"])
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--per-category", type=int, default=10)
    ap.add_argument("--max-conclusions", type=int, default=None,
                    help="keep only cases with at most this many conclusions (use the same "
                         "value for export, run and score)")
    ap.add_argument("--conditions", nargs="+", choices=CONDS, default=list(CONDS),
                    help="which prompts to run or export (default: all three)")
    ap.add_argument("--append", action="store_true",
                    help="add to an existing responses file, skipping what is already there")
    ap.add_argument("--overwrite", action="store_true", help="replace an existing responses file")
    ap.add_argument("--redo", action="store_true",
                    help="re-run the selected --conditions, replacing only those responses and "
                         "keeping the rest of the file")
    ap.add_argument("--prompts", default="prompts.jsonl")
    ap.add_argument("--responses", default="responses.jsonl")
    ap.add_argument("--cmd", help="local command that reads a prompt on stdin and prints the reply")
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--label", default="model")
    args = ap.parse_args()
    cases = build_cases(args.seed, args.per_category, args.max_conclusions)
    if not cases:
        sys.exit("no cases match these settings")

    if args.command == "demo":
        print(f"{len(cases)} cases, seed {args.seed}. Dummy models only, to prove the pipeline.")
        report(cases, dummy_responses(cases, "perfect"), "dummy: derives and revises perfectly")
        report(cases, dummy_responses(cases, "stale"),
               "dummy: answers from scratch, never revises, ignores the hint")
        report(cases, dummy_responses(cases, "assisted"),
               "dummy: answers from scratch, never revises unaided, revises with the hint")
        return

    if args.command == "export":
        n = 0
        with open(args.prompts, "w", encoding="utf-8") as f:
            for c in cases:
                for cond in args.conditions:
                    f.write(json.dumps({"id": c.id, "condition": cond,
                                        "prompt": build_prompt(c, cond)}) + "\n")
                    n += 1
        print(f"wrote {n} prompts to {args.prompts}")
        return

    if args.command == "run":
        if not args.cmd:
            sys.exit("run needs --cmd, e.g. --cmd \"ollama run llama3\"")
        exists = os.path.exists(args.responses)
        if exists and args.redo:
            keep = {k: v for k, v in read_responses(args.responses).items()
                    if k[1] not in args.conditions}
            with open(args.responses, "w", encoding="utf-8") as f:
                for (cid, cnd), text in keep.items():
                    f.write(json.dumps({"id": cid, "condition": cnd, "response": text}) + "\n")
            args.append = True
        if exists and not (args.append or args.overwrite):
            sys.exit(f"{args.responses} already exists. Use --append to add to it or resume, "
                     f"--redo to re-run chosen conditions, or --overwrite to replace it.")
        done = read_responses(args.responses) if (exists and args.append) else {}
        todo = [(c, cond) for c in cases for cond in args.conditions if (c.id, cond) not in done]
        print(f"{len(todo)} prompts to run ({len(done)} already in {args.responses})")
        argv = shlex.split(args.cmd)
        with open(args.responses, "a" if args.append else "w", encoding="utf-8") as f:
            for i, (c, cond) in enumerate(todo, 1):
                p = subprocess.run(argv, input=build_prompt(c, cond), capture_output=True,
                                   text=True, encoding="utf-8", errors="replace",
                                   timeout=args.timeout)
                if p.returncode != 0:
                    sys.exit(f"command failed on {c.id}/{cond} (exit {p.returncode}):\n"
                             f"{p.stderr.strip()[:500]}")
                f.write(json.dumps({"id": c.id, "condition": cond,
                                    "response": p.stdout}) + "\n")
                f.flush()
                print(f"  {i}/{len(todo)} prompts done", end="\r")
        print(f"\nwrote {args.responses}")
        report(cases, read_responses(args.responses), args.label)
        return

    report(cases, read_responses(args.responses), args.label)


if __name__ == "__main__":
    main()
