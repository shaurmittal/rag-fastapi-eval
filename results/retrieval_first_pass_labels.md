# Ablation results

51 questions per configuration. Labels: first-pass only.

## Mean scores

| config | hit@1 | hit@5 | recall@5 | recall@10 | precision@5 | mrr@10 |
|---|---|---|---|---|---|---|
| naive+vector | 0.392 | 0.667 | 0.667 | 0.735 | 0.306 | 0.517 |
| naive+rerank | 0.431 | 0.667 | 0.667 | 0.794 | 0.294 | 0.533 |
| structured+vector | 0.431 | 0.627 | 0.618 | 0.755 | 0.345 | 0.519 |
| structured+rerank | 0.431 | 0.686 | 0.676 | 0.794 | 0.341 | 0.542 |

## docs-answerable questions only (n=45)

| config | hit@1 | hit@5 | recall@5 | recall@10 | precision@5 | mrr@10 |
|---|---|---|---|---|---|---|
| naive+vector | 0.422 | 0.667 | 0.667 | 0.744 | 0.320 | 0.536 |
| naive+rerank | 0.489 | 0.689 | 0.689 | 0.767 | 0.320 | 0.575 |
| structured+vector | 0.444 | 0.622 | 0.611 | 0.722 | 0.329 | 0.525 |
| structured+rerank | 0.467 | 0.667 | 0.656 | 0.789 | 0.329 | 0.566 |

## code-answerable questions only (n=6)

| config | hit@1 | hit@5 | recall@5 | recall@10 | precision@5 | mrr@10 |
|---|---|---|---|---|---|---|
| naive+vector | 0.167 | 0.667 | 0.667 | 0.667 | 0.200 | 0.375 |
| naive+rerank | 0.000 | 0.500 | 0.500 | 1.000 | 0.100 | 0.221 |
| structured+vector | 0.333 | 0.667 | 0.667 | 1.000 | 0.467 | 0.471 |
| structured+rerank | 0.167 | 0.833 | 0.833 | 0.833 | 0.433 | 0.358 |

## Paired comparisons (95% bootstrap CI of the mean difference)

A difference is only supported by the data if its interval excludes 0.

| factor | metric | Δ mean | 95% CI | win/loss/tie | verdict |
|---|---|---|---|---|---|
| chunking, no rerank | hit@1 | +0.039 | [-0.078, +0.157] | 6/4/41 | no clear difference |
| chunking, no rerank | hit@5 | -0.039 | [-0.176, +0.098] | 5/7/39 | no clear difference |
| chunking, no rerank | recall@5 | -0.049 | [-0.176, +0.078] | 5/7/39 | no clear difference |
| chunking, no rerank | recall@10 | +0.020 | [-0.039, +0.098] | 2/1/48 | no clear difference |
| chunking, no rerank | precision@5 | +0.039 | [-0.031, +0.114] | 18/16/17 | no clear difference |
| chunking, no rerank | mrr@10 | +0.002 | [-0.087, +0.092] | 10/11/30 | no clear difference |
| chunking, with rerank | hit@1 | +0.000 | [-0.118, +0.118] | 5/5/41 | no clear difference |
| chunking, with rerank | hit@5 | +0.020 | [-0.098, +0.137] | 6/5/40 | no clear difference |
| chunking, with rerank | recall@5 | +0.010 | [-0.118, +0.127] | 6/5/40 | no clear difference |
| chunking, with rerank | recall@10 | +0.000 | [-0.098, +0.098] | 3/3/45 | no clear difference |
| chunking, with rerank | precision@5 | +0.047 | [-0.035, +0.133] | 16/9/26 | no clear difference |
| chunking, with rerank | mrr@10 | +0.009 | [-0.084, +0.100] | 16/11/24 | no clear difference |
| reranking, naive chunks | hit@1 | +0.039 | [-0.098, +0.176] | 7/5/39 | no clear difference |
| reranking, naive chunks | hit@5 | +0.000 | [-0.098, +0.098] | 4/4/43 | no clear difference |
| reranking, naive chunks | recall@5 | +0.000 | [-0.098, +0.098] | 4/4/43 | no clear difference |
| reranking, naive chunks | recall@10 | +0.059 | [-0.020, +0.137] | 4/1/46 | no clear difference |
| reranking, naive chunks | precision@5 | -0.012 | [-0.071, +0.051] | 10/14/27 | no clear difference |
| reranking, naive chunks | mrr@10 | +0.016 | [-0.072, +0.106] | 14/11/26 | no clear difference |
| reranking, structured chunks | hit@1 | +0.000 | [-0.118, +0.118] | 4/4/43 | no clear difference |
| reranking, structured chunks | hit@5 | +0.059 | [-0.039, +0.157] | 5/2/44 | no clear difference |
| reranking, structured chunks | recall@5 | +0.059 | [-0.039, +0.157] | 5/2/44 | no clear difference |
| reranking, structured chunks | recall@10 | +0.039 | [-0.039, +0.118] | 3/1/47 | no clear difference |
| reranking, structured chunks | precision@5 | -0.004 | [-0.067, +0.055] | 12/11/28 | no clear difference |
| reranking, structured chunks | mrr@10 | +0.023 | [-0.055, +0.100] | 14/9/28 | no clear difference |

## Latency per query (seconds)

| config | rerank | retrieve |
|---|---|---|
| naive+vector | - | p50 0.061 / p95 0.118 |
| naive+rerank | p50 0.965 / p95 1.013 | p50 0.065 / p95 0.118 |
| structured+vector | - | p50 0.059 / p95 0.113 |
| structured+rerank | p50 0.964 / p95 0.975 | p50 0.065 / p95 0.118 |
