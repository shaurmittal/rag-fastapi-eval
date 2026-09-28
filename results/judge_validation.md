# Judge validation

Output of `uv run python -m evaluation.judge_validation report` on the fixed random sample in `data/judge_validation.jsonl` (60 claims, 30 answers).
Raters: `human` = the author, blind to the judge (random subset of 12 claims + 8 answers); `llm_rater` = an independent pass by Claude Opus 5.5, blind to the judge and the human labels (all 90); `judge_retest` = the judge (Claude Sonnet 5) re-scoring the same items.


One human claim label (item 11) was corrected on re-review after unblinding: the supporting sentence is quoted near-verbatim in the context.

```

## claim (n=60 sampled)
judge vs human      n=12  agreement=0.75  kappa=0.25  (human yes 9/12, judge yes 10/12; judge stricter on 1, more lenient on 2)
judge vs llm_rater  n=60  agreement=0.95  kappa=0.89  (llm_rater yes 40/60, judge yes 39/60; judge stricter on 2, more lenient on 1)
human vs llm_rater  n=12  agreement=0.83  kappa=0.56  (llm_rater yes 9/12, human yes 9/12; human stricter on 1, more lenient on 1)
judge vs judge_retest n=60  agreement=0.95  kappa=0.89  (judge_retest yes 42/60, judge yes 39/60; judge stricter on 3, more lenient on 0)

## correctness (n=30 sampled)
judge vs human      n=8   agreement=1.00  kappa=1.00  (human yes 6/8, judge yes 6/8; judge stricter on 0, more lenient on 0)
judge vs llm_rater  n=30  agreement=0.97  kappa=0.91  (llm_rater yes 22/30, judge yes 23/30; judge stricter on 0, more lenient on 1)
human vs llm_rater  n=8   agreement=1.00  kappa=1.00  (llm_rater yes 6/8, human yes 6/8; human stricter on 0, more lenient on 0)
judge vs judge_retest n=30  agreement=1.00  kappa=1.00  (judge_retest yes 23/30, judge yes 23/30; judge stricter on 0, more lenient on 0)
```
