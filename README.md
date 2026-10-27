# RevisionTrace

**A pure-Python benchmark for belief revision: when new evidence invalidates a stored conclusion, does the correction reach every conclusion that depended on it, and leave the rest alone?**

![Python 3.13](https://img.shields.io/badge/python-3.13-blue) ![License: MIT](https://img.shields.io/badge/license-MIT-green)

Most evaluations check whether a system can answer from the current evidence. This repo checks what happens next: new evidence arrives, and the system has to fix the conclusions it already stored.

Read the full write-up on Towards Data Science → **[We Taught AI to Answer Questions. Who Taught It to Change Its Mind?](https://towardsdatascience.com/author/emmimalp.alexander/)**

Standard library only. No API keys, no network calls, no third-party packages.

## Why this exists

A system can derive the right answer from the current evidence and still end up in the wrong state when it has to update conclusions it formed earlier. Think of a travel assistant. The airline cancels a flight. Does the hotel check-in change? Does the pickup get cancelled? Does the "trip confirmed" flag flip? Does the dentist appointment stay exactly where it was?

RevisionTrace makes that question measurable. It reaches the same final state by two routes, using the same evidence, the same rules and an independent oracle:

1. **Answer from scratch.** The system gets the final evidence and derives every conclusion.
2. **Revise from an old state.** The system starts from stored conclusions, receives evidence updates, and has to fix what depended on them.

## Headline result

Exact final-state agreement over 300 seeded cases (seed 2026):

| Strategy | Answer from scratch | Revise from old state |
|---|---|---|
| `full_recompute` | 100% | 100% |
| `incremental` | 100% | 100% |
| `history_shallow` | 100% | 59.0% |
| `history_sticky` | 100% | 57.0% |
| `history_branch_first` | 100% | 29.0% |
| `history_flush` | 100% | 0.7% |

**Read this before quoting the numbers.** These are controlled strategies that I wrote on purpose. Four of them are failure models, each breaking one rule. They are not measurements of language models, and the rates say nothing about how often a real AI system fails. The benchmark tests revision *algorithms* under a defined logic.

## Quick start

You need Python 3 and nothing else. It was tested on Python 3.13. Older versions are untested.

```bash
git clone https://github.com/Emmimal/revisiontrace.git
cd revisiontrace

python revisiontrace.py --selftest      # unit checks only
python revisiontrace.py                 # the full benchmark, about a second
```

Useful options:

```bash
python revisiontrace.py --seed 7 --cases 100            # a different sample (100 per category)
python revisiontrace.py --seed 42 --cases 200 --perms 10 --out my_run
```

The run writes its files to `revisiontrace_out/` (or the folder you pass to `--out`).

### Check that your run matches mine

The first lines of the output should read:

```
selftest: ok
verified: full_recompute and incremental match the oracle on all 300 cases
verified: all 6 strategies answer correctly from scratch on all 300 cases

RevisionTrace  |  300 cases  |  case-set hash 506be3856003
```

The same seed gives the same cases and byte-identical result files. The full console output of my seed 2026 run is in [`results/seed2026/expected_output.txt`](results/seed2026/expected_output.txt).

## Project structure

```
revisiontrace/
├── revisiontrace.py      # graph, evidence, oracle, strategies, generator, scorer, report
├── llm_eval.py           # optional local language model arm
├── figures/              # the article diagrams as SVG
└── results/seed2026/     # reference outputs: console output, CSV files, failure traces
```

A run of `revisiontrace.py` writes: `results_summary.csv`, `ablation.csv`, `per_case.csv`, `cases.json`, `failure_traces.txt`, `featured_stale.svg` and `featured_collateral.svg`.

## What it models

![A small dependency graph](figures/structure.svg)

**Graph.** Facts feed conclusions. Conclusions feed other conclusions. A conclusion that nothing else depends on is a *decision*. Each conclusion uses one rule: `AND`, `OR`, or `k-of-n`. Every graph also carries a small independent branch, so a change in one place can be checked for disturbing another. A typical case has about 12 conclusions (4 to 22) built on about 10 facts, with chains two to five conclusions deep.

**Three-valued logic.** Every node is `TRUE`, `FALSE` or `UNKNOWN` (Kleene's strong three-valued logic). A k-of-n rule with `t` TRUE inputs and `u` UNKNOWN inputs is TRUE if `t >= k`, FALSE if `t + u < k`, and otherwise UNKNOWN. AND is n-of-n and OR is 1-of-n.

**Evidence is a set of claims.** An update is either `ADD claim_id fact value rank` or `RETRACT claim_id`. A fact takes the value of its highest-ranked live claims. If those claims disagree, it is `UNKNOWN`. With no live claims it is also `UNKNOWN`. A retraction is permanent for that claim id, even if it arrives before the claim it cancels. Because of this, the correct final state depends only on the final set of claims and not on the order updates arrive in. That is what makes the update-order test fair.

**Oracle.** It computes the expected state straight from the final evidence, with explicit Kleene operators and a "k-th best input" form of the threshold. It shares no update logic with the strategies. A self-test compares it with the engine's formulation across 5,000 random rule evaluations before every run.

## Strategies

| Strategy | What it does | Type |
|---|---|---|
| `full_recompute` | Recomputes every node after every update | Correct by design |
| `incremental` | Re-evaluates only the dependents of a changed node and stops where values stop changing | Correct by design |
| `history_shallow` | Updates the changed fact and its direct dependents only | Failure model |
| `history_branch_first` | Cascades, but follows only the first dependent of each node | Failure model |
| `history_sticky` | Cascades, but never retreats from a decided belief to UNKNOWN | Failure model |
| `history_flush` | Wipes every conclusion to UNKNOWN, then refills only the descendants of the change | Failure model |

The run asserts that `full_recompute` and `incremental` match the oracle on every case, and that all six strategies answer correctly from scratch. The `history_*` strategies start from the correct stored state and then revise it badly on purpose. They are analogies for real mistakes, not claims about any product.

## Metrics

Each conclusion is compared three ways: what it held before, what the oracle expects, and what the system holds after the updates.

| Metric | Meaning |
|---|---|
| **stale** | Should have changed, still holds the old value |
| **collateral** | Should have stayed the same, but changed |
| **wrong direction** | Changed, but to the wrong value |
| **false resolution** | Should be UNKNOWN, but the system committed to TRUE or FALSE |
| **hidden damage** | A failed case where every final decision is still correct and only interior conclusions are wrong |
| **exact** | Every conclusion in the case matches the oracle (95% Wilson interval) |
| **order stable / order correct** | Cases with two or more updates are replayed under extra random orderings. Stable means every ordering ends in the same state. Correct means every ordering matches the oracle |
| **cost** | Node evaluations per update. This is not wall-clock time |

## Test categories

300 cases, 50 per category: `single_fact_change`, `multi_support`, `shared_dependency`, `conflicting_evidence`, `unrelated_change` and `multi_update` (four to eight updates, some added and later retracted). Each case is checked at generation time to confirm it really exercises its category.

## Main findings (seed 2026)

- **Answering and revising can come apart.** Every strategy answers correctly from scratch. Revising ranges from 100% down to 0.7%.
- **Stale and collateral are different failures.** Three strategies leave 33% to 65% of the conclusions that should change untouched. Flushing damages 58% of the conclusions that should stay.
- **Failures can hide.** For the three stale-prone strategies, 16% to 23% of failed cases still had every final decision correct.
- **Consistent is not the same as correct.** The sticky strategy ended in the same state under every reordering in 98% of multi-update cases, yet that state was right in only 75%.
- **What makes revision work.** An explicit record of what depends on what, plus a rule to propagate until values stop changing. In the propagation-limit ablation, exact agreement is 8.0% at zero hops, 59.0% at one, 90.3% at two, 99.3% at three and 100% at four. Incremental revision used about a quarter of the node evaluations of full recomputation (5.2 against 21.9 per update).

![Exact final-state agreement](figures/agreement.svg)

### Robustness

I re-ran the benchmark on three other seeds (7, 42 and 2027). The two correct strategies stayed at 100%. Exact agreement when revising ranged from 55.3% to 60.7% for shallow, 57.0% to 60.7% for sticky, 25.7% to 33.3% for branch-first and 0.7% to 1.3% for flush. A larger validation run (`--seed 42 --cases 200 --perms 10`, 1,200 cases) again matched the oracle on every case for both correct strategies. The case set for a given seed is deterministic.

## Optional: a local language model arm

`llm_eval.py` is a separate experiment and should be reported separately. It needs no API. It writes prompts, optionally pipes them into a local program you name (such as Ollama), and scores the replies against the same oracle.

```bash
python llm_eval.py demo                                  # checks the pipeline with two dummy models
python llm_eval.py export --per-category 10              # writes prompts.jsonl to paste into any model
python llm_eval.py run --cmd "ollama run mymodel" --label mymodel --max-conclusions 8
python llm_eval.py score --responses responses.jsonl --label mymodel
```

Three prompt conditions use identical rules:

- `answer`: the final evidence only. Tests derivation from scratch.
- `revise`: the earlier evidence, the model's stored conclusions, then the updates.
- `revise_hint`: the same as `revise`, plus a dependency record that lists which conclusions to re-derive.

The score table also shows node accuracy (the share of individual conclusions right) and a **copy-stored baseline** that never revises. The `revise_hint+carry` row re-scores the hint condition after filling any unlisted conclusion with its stored value. Report it next to the strict row, never instead of it.

Other options: `--max-conclusions N` keeps only small graphs, `--append` adds conditions to an existing responses file, `--overwrite` replaces it, `--redo` re-asks conditions that already have responses, and `--timeout` and `--seed` do what they say. Paired lines such as "right on answer, wrong on revise" can rest on very few cases. Do not quote them when the count is small.

### One small run, reported as a limit

I ran Gemma 3 4B through Ollama (temperature 0, seed 42, context 8192) on the 66 cases with eight or fewer conclusions (`--per-category 50 --max-conclusions 8`).

| Row (66 cases) | Exact | Conclusions right | Stale | Collateral | Wrong direction |
|---|---|---|---|---|---|
| Copy stored, never revise | 12.1% | 79.4% | 100% | 0% | 0 |
| Revise | 16.7% | 77.3% | 81.4% | 7.0% | 2 |
| Revise with hint (carry) | 13.6% | 77.1% | 72.2% | 7.0% | 12 |
| Revise with hint (strict) | 7.6% | 52.0% | 72.2% | 38.5% | 12 |

From scratch the model got 58.6% of conclusions right. No model row beats the copy baseline on conclusions right. The hint changed the kind of error more than the amount. This is one run of one small model on a subset of small cases, and it does not separate revision from derivation. It is not evidence about language models in general. Raw responses are not included in this repository.

## When to use this

Worth it when you:

- store derived conclusions, such as summaries, cached answers, agent notes or status flags, and want to test how they behave when evidence changes
- want a small, deterministic test bed for a revision or invalidation strategy
- want to see stale conclusions and collateral damage measured separately

Skip it when you:

- need a measurement of how a particular language model or product behaves on real text
- want realistic, ambiguous evidence. The logic here is small and symbolic

## Known limitations

- **It is not an evaluation of language models.** The six strategies are code. The 100% in the "answer from scratch" column is true by construction.
- **The failure models are my constructions.** Their rates follow from how I built them. I did not measure how often any real agent behaves like one of them.
- **The logic is small and symbolic.** Real evidence is ambiguous. Extracting a dependency record from text is a hard problem that this benchmark assumes away. The incremental strategy starts from an oracle-correct state.
- **The sample is modest.** There are 300 cases. Each depth group has 72 to 78. Wilson intervals are reported, and overlapping groups should not be over-read.
- **Cost is counted in node evaluations.** I did not time anything, so fewer evaluations do not guarantee faster execution.

## Related work

- Wilie et al. (2024), *Belief Revision: The Adaptability of Large Language Models Reasoning*, EMNLP 2024. https://aclanthology.org/2024.emnlp-main.586
- Zhong et al. (2023), *MQuAKE: Assessing Knowledge Editing in Language Models via Multi-Hop Questions*, EMNLP 2023. https://aclanthology.org/2023.emnlp-main.971
- Cohen et al. (2024), *Evaluating the Ripple Effects of Knowledge Editing in Language Models*, TACL 12, 283-298. https://aclanthology.org/2024.tacl-1.16/
- Jenkins (2026), *AGM-Bench: Do Large Language Models Revise Beliefs Rationally?*, ICLR 2026 Workshop on Logical Reasoning of LLMs (poster). https://www.iclr.cc/virtual/2026/10017503
- Alchourrón, Gärdenfors and Makinson (1985), *On the logic of theory change*, Journal of Symbolic Logic 50, 510-530. https://doi.org/10.2307/2274239
- Doyle (1979), *A truth maintenance system*, Artificial Intelligence 12, 231-272.
- Gupta, Mumick and Subrahmanian (1993), *Maintaining views incrementally*, ACM SIGMOD, 157-166. https://dl.acm.org/doi/10.1145/170035.170066
- Marcos, Přenosil and Égré, *Many-valued logic*, Stanford Encyclopedia of Philosophy (Spring 2026 edition). https://plato.stanford.edu/archives/spr2026/entries/logic-manyvalued/
- Wilson (1927), *Probable inference, the law of succession, and statistical inference*, Journal of the American Statistical Association 22, 209-212.
- Gemma Team, Google DeepMind (2025), *Gemma 3 technical report*. https://arxiv.org/abs/2503.19786
- Ollama. https://github.com/ollama/ollama

I describe these works in my own words. No text, data or figures are reproduced from them.

## Citation

If you use RevisionTrace, please cite the repository:

```
Alexander, E. P. (2026). RevisionTrace: a pure-Python benchmark for belief revision
in dependent conclusions. GitHub. https://github.com/Emmimal/revisiontrace
```

## License

MIT. See [LICENSE](LICENSE).
