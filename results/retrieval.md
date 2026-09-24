# Ablation results

51 questions per configuration. Labels: pooled (first-pass + pooled-review additions).

## Mean scores

| config | hit@1 | hit@5 | recall@5 | recall@10 | precision@5 | mrr@10 |
|---|---|---|---|---|---|---|
| naive+vector | 0.490 | 0.765 | 0.755 | 0.794 | 0.408 | 0.603 |
| naive+rerank | 0.471 | 0.765 | 0.755 | 0.833 | 0.365 | 0.586 |
| structured+vector | 0.471 | 0.706 | 0.696 | 0.794 | 0.443 | 0.577 |
| structured+rerank | 0.510 | 0.765 | 0.755 | 0.814 | 0.443 | 0.620 |

## docs-answerable questions only (n=45)

| config | hit@1 | hit@5 | recall@5 | recall@10 | precision@5 | mrr@10 |
|---|---|---|---|---|---|---|
| naive+vector | 0.511 | 0.756 | 0.744 | 0.789 | 0.427 | 0.611 |
| naive+rerank | 0.533 | 0.800 | 0.789 | 0.811 | 0.396 | 0.632 |
| structured+vector | 0.489 | 0.689 | 0.678 | 0.767 | 0.427 | 0.582 |
| structured+rerank | 0.556 | 0.756 | 0.744 | 0.811 | 0.440 | 0.650 |

## code-answerable questions only (n=6)

| config | hit@1 | hit@5 | recall@5 | recall@10 | precision@5 | mrr@10 |
|---|---|---|---|---|---|---|
| naive+vector | 0.333 | 0.833 | 0.833 | 0.833 | 0.267 | 0.542 |
| naive+rerank | 0.000 | 0.500 | 0.500 | 1.000 | 0.133 | 0.244 |
| structured+vector | 0.333 | 0.833 | 0.833 | 1.000 | 0.567 | 0.544 |
| structured+rerank | 0.167 | 0.833 | 0.833 | 0.833 | 0.467 | 0.400 |

## Paired comparisons (95% bootstrap CI of the mean difference)

A difference is only supported by the data if its interval excludes 0.

| factor | metric | Δ mean | 95% CI | win/loss/tie | verdict |
|---|---|---|---|---|---|
| chunking, no rerank | hit@1 | -0.020 | [-0.137, +0.098] | 5/6/40 | no clear difference |
| chunking, no rerank | hit@5 | -0.059 | [-0.157, +0.039] | 2/5/44 | no clear difference |
| chunking, no rerank | recall@5 | -0.059 | [-0.157, +0.039] | 2/5/44 | no clear difference |
| chunking, no rerank | recall@10 | +0.000 | [-0.078, +0.078] | 2/2/47 | no clear difference |
| chunking, no rerank | precision@5 | +0.035 | [-0.043, +0.122] | 17/18/16 | no clear difference |
| chunking, no rerank | mrr@10 | -0.026 | [-0.113, +0.065] | 8/12/31 | no clear difference |
| chunking, with rerank | hit@1 | +0.039 | [-0.098, +0.176] | 7/5/39 | no clear difference |
| chunking, with rerank | hit@5 | +0.000 | [-0.098, +0.098] | 4/4/43 | no clear difference |
| chunking, with rerank | recall@5 | +0.000 | [-0.098, +0.098] | 4/4/43 | no clear difference |
| chunking, with rerank | recall@10 | -0.020 | [-0.098, +0.059] | 2/3/46 | no clear difference |
| chunking, with rerank | precision@5 | +0.078 | [-0.000, +0.161] | 17/11/23 | no clear difference |
| chunking, with rerank | mrr@10 | +0.034 | [-0.063, +0.134] | 15/10/26 | no clear difference |
| reranking, naive chunks | hit@1 | -0.020 | [-0.157, +0.118] | 6/7/38 | no clear difference |
| reranking, naive chunks | hit@5 | +0.000 | [-0.078, +0.078] | 2/2/47 | no clear difference |
| reranking, naive chunks | recall@5 | +0.000 | [-0.078, +0.078] | 2/2/47 | no clear difference |
| reranking, naive chunks | recall@10 | +0.039 | [+0.000, +0.098] | 2/0/49 | no clear difference |
| reranking, naive chunks | precision@5 | -0.043 | [-0.098, +0.012] | 11/17/23 | no clear difference |
| reranking, naive chunks | mrr@10 | -0.017 | [-0.107, +0.070] | 12/11/28 | no clear difference |
| reranking, structured chunks | hit@1 | +0.039 | [-0.078, +0.157] | 6/4/41 | no clear difference |
| reranking, structured chunks | hit@5 | +0.059 | [+0.000, +0.137] | 3/0/48 | no clear difference |
| reranking, structured chunks | recall@5 | +0.059 | [+0.000, +0.137] | 3/0/48 | no clear difference |
| reranking, structured chunks | recall@10 | +0.020 | [-0.039, +0.078] | 2/1/48 | no clear difference |
| reranking, structured chunks | precision@5 | +0.000 | [-0.055, +0.055] | 10/10/31 | no clear difference |
| reranking, structured chunks | mrr@10 | +0.043 | [-0.032, +0.122] | 14/9/28 | no clear difference |

## Latency per query (seconds)

| config | rerank | retrieve |
|---|---|---|
| naive+vector | - | p50 0.060 / p95 0.114 |
| naive+rerank | p50 0.970 / p95 1.063 | p50 0.065 / p95 0.124 |
| structured+vector | - | p50 0.058 / p95 0.116 |
| structured+rerank | p50 0.984 / p95 1.042 | p50 0.065 / p95 0.119 |
