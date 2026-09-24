"""Statistics for comparing configurations on a small eval set.

With ~50 questions, a difference of a few points between two configs can
easily be noise. Two tools make comparisons honest:

PAIRED COMPARISON
    Every config answers the SAME questions, so compare them question by
    question (wins / losses / ties) rather than comparing two averages.
    Pairing removes question difficulty as a source of variance: a hard
    question drags both configs down together instead of looking like a
    difference between them.

PAIRED BOOTSTRAP CONFIDENCE INTERVAL
    Resample the questions with replacement many times, recompute the mean
    difference each time, and read the 2.5th and 97.5th percentiles. If the
    95% interval contains 0, the data don't support claiming a difference.
    No distributional assumptions - it uses the observed per-question
    differences directly.
"""

import random


def paired_record(a, b):
    """Wins/losses/ties of config A over config B on one metric."""
    wins = sum(x > y for x, y in zip(a, b))
    losses = sum(x < y for x, y in zip(a, b))
    return wins, losses, len(a) - wins - losses


def bootstrap_ci(a, b, n_resamples=10_000, seed=0):
    """95% CI for mean(a - b), resampling questions with replacement."""
    diffs = [x - y for x, y in zip(a, b)]
    rng = random.Random(seed)  # fixed seed: the same data always gives the same interval
    n = len(diffs)
    means = sorted(sum(rng.choices(diffs, k=n)) / n for _ in range(n_resamples))
    return means[int(0.025 * n_resamples)], means[int(0.975 * n_resamples)]


def mean(values):
    return sum(values) / len(values) if values else float("nan")


def percentile(values, p):
    ordered = sorted(values)
    return ordered[min(int(p / 100 * len(ordered)), len(ordered) - 1)]
