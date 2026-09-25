# Ablation results

51 questions per configuration. Labels: pooled (first-pass + pooled-review additions).

## Mean scores

| config | hit@1 | hit@5 | recall@5 | recall@10 | precision@5 | mrr@10 | faithfulness | relevance | correctness | has_citation | cited_source_precision |
|---|---|---|---|---|---|---|---|---|---|---|---|
| naive+vector | 0.490 | 0.765 | 0.755 | 0.794 | 0.408 | 0.603 | 0.609 | 0.961 | 0.725 | 1.000 | 0.563 |
| naive+rerank | 0.471 | 0.765 | 0.755 | 0.833 | 0.365 | 0.586 | 0.628 | 0.922 | 0.647 | 1.000 | 0.526 |
| structured+vector | 0.471 | 0.706 | 0.696 | 0.794 | 0.443 | 0.577 | 0.677 | 0.882 | 0.686 | 1.000 | 0.541 |
| structured+rerank | 0.510 | 0.765 | 0.755 | 0.814 | 0.443 | 0.620 | 0.641 | 0.882 | 0.686 | 0.980 | 0.594 |

## docs-answerable questions only (n=45)

| config | hit@1 | hit@5 | recall@5 | recall@10 | precision@5 | mrr@10 | faithfulness | relevance | correctness | has_citation | cited_source_precision |
|---|---|---|---|---|---|---|---|---|---|---|---|
| naive+vector | 0.511 | 0.756 | 0.744 | 0.789 | 0.427 | 0.611 | 0.613 | 0.956 | 0.733 | 1.000 | 0.590 |
| naive+rerank | 0.533 | 0.800 | 0.789 | 0.811 | 0.396 | 0.632 | 0.644 | 0.911 | 0.667 | 1.000 | 0.551 |
| structured+vector | 0.489 | 0.689 | 0.678 | 0.767 | 0.427 | 0.582 | 0.685 | 0.867 | 0.667 | 1.000 | 0.526 |
| structured+rerank | 0.556 | 0.756 | 0.744 | 0.811 | 0.440 | 0.650 | 0.656 | 0.867 | 0.689 | 1.000 | 0.586 |

## code-answerable questions only (n=6)

| config | hit@1 | hit@5 | recall@5 | recall@10 | precision@5 | mrr@10 | faithfulness | relevance | correctness | has_citation | cited_source_precision |
|---|---|---|---|---|---|---|---|---|---|---|---|
| naive+vector | 0.333 | 0.833 | 0.833 | 0.833 | 0.267 | 0.542 | 0.577 | 1.000 | 0.667 | 1.000 | 0.360 |
| naive+rerank | 0.000 | 0.500 | 0.500 | 1.000 | 0.133 | 0.244 | 0.510 | 1.000 | 0.500 | 1.000 | 0.339 |
| structured+vector | 0.333 | 0.833 | 0.833 | 1.000 | 0.567 | 0.544 | 0.615 | 1.000 | 0.833 | 1.000 | 0.653 |
| structured+rerank | 0.167 | 0.833 | 0.833 | 0.833 | 0.467 | 0.400 | 0.534 | 1.000 | 0.667 | 0.833 | 0.667 |

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
| chunking, no rerank | faithfulness | +0.068 | [-0.004, +0.140] | 30/20/1 | no clear difference |
| chunking, no rerank | relevance | -0.078 | [-0.176, +0.000] | 1/5/45 | no clear difference |
| chunking, no rerank | correctness | -0.039 | [-0.196, +0.118] | 7/9/35 | no clear difference |
| chunking, no rerank | has_citation | +0.000 | [+0.000, +0.000] | 0/0/51 | no clear difference |
| chunking, no rerank | cited_source_precision | -0.022 | [-0.129, +0.085] | 10/15/26 | no clear difference |
| chunking, with rerank | hit@1 | +0.039 | [-0.098, +0.176] | 7/5/39 | no clear difference |
| chunking, with rerank | hit@5 | +0.000 | [-0.098, +0.098] | 4/4/43 | no clear difference |
| chunking, with rerank | recall@5 | +0.000 | [-0.098, +0.098] | 4/4/43 | no clear difference |
| chunking, with rerank | recall@10 | -0.020 | [-0.098, +0.059] | 2/3/46 | no clear difference |
| chunking, with rerank | precision@5 | +0.078 | [-0.000, +0.161] | 17/11/23 | no clear difference |
| chunking, with rerank | mrr@10 | +0.034 | [-0.063, +0.134] | 15/10/26 | no clear difference |
| chunking, with rerank | faithfulness | +0.013 | [-0.050, +0.076] | 23/24/3 | no clear difference |
| chunking, with rerank | relevance | -0.039 | [-0.118, +0.039] | 1/3/47 | no clear difference |
| chunking, with rerank | correctness | +0.039 | [-0.078, +0.137] | 5/3/43 | no clear difference |
| chunking, with rerank | has_citation | -0.020 | [-0.059, +0.000] | 0/1/50 | no clear difference |
| chunking, with rerank | cited_source_precision | +0.062 | [-0.041, +0.166] | 18/15/17 | no clear difference |
| reranking, naive chunks | hit@1 | -0.020 | [-0.157, +0.118] | 6/7/38 | no clear difference |
| reranking, naive chunks | hit@5 | +0.000 | [-0.078, +0.078] | 2/2/47 | no clear difference |
| reranking, naive chunks | recall@5 | +0.000 | [-0.078, +0.078] | 2/2/47 | no clear difference |
| reranking, naive chunks | recall@10 | +0.039 | [+0.000, +0.098] | 2/0/49 | no clear difference |
| reranking, naive chunks | precision@5 | -0.043 | [-0.098, +0.012] | 11/17/23 | no clear difference |
| reranking, naive chunks | mrr@10 | -0.017 | [-0.107, +0.070] | 12/11/28 | no clear difference |
| reranking, naive chunks | faithfulness | +0.017 | [-0.043, +0.080] | 23/23/4 | no clear difference |
| reranking, naive chunks | relevance | -0.039 | [-0.118, +0.039] | 1/3/47 | no clear difference |
| reranking, naive chunks | correctness | -0.078 | [-0.196, +0.039] | 3/7/41 | no clear difference |
| reranking, naive chunks | has_citation | +0.000 | [+0.000, +0.000] | 0/0/51 | no clear difference |
| reranking, naive chunks | cited_source_precision | -0.037 | [-0.125, +0.053] | 11/17/23 | no clear difference |
| reranking, structured chunks | hit@1 | +0.039 | [-0.078, +0.157] | 6/4/41 | no clear difference |
| reranking, structured chunks | hit@5 | +0.059 | [+0.000, +0.137] | 3/0/48 | no clear difference |
| reranking, structured chunks | recall@5 | +0.059 | [+0.000, +0.137] | 3/0/48 | no clear difference |
| reranking, structured chunks | recall@10 | +0.020 | [-0.039, +0.078] | 2/1/48 | no clear difference |
| reranking, structured chunks | precision@5 | +0.000 | [-0.055, +0.055] | 10/10/31 | no clear difference |
| reranking, structured chunks | mrr@10 | +0.043 | [-0.032, +0.122] | 14/9/28 | no clear difference |
| reranking, structured chunks | faithfulness | -0.035 | [-0.098, +0.026] | 21/28/2 | no clear difference |
| reranking, structured chunks | relevance | +0.000 | [-0.078, +0.078] | 2/2/47 | no clear difference |
| reranking, structured chunks | correctness | +0.000 | [-0.078, +0.078] | 2/2/47 | no clear difference |
| reranking, structured chunks | has_citation | -0.020 | [-0.059, +0.000] | 0/1/50 | no clear difference |
| reranking, structured chunks | cited_source_precision | +0.063 | [-0.010, +0.145] | 11/11/28 | no clear difference |

## Latency per query (seconds)

| config | rerank | retrieve |
|---|---|---|
| naive+vector | - | p50 0.062 / p95 0.122 |
| naive+rerank | p50 0.959 / p95 1.153 | p50 0.065 / p95 0.120 |
| structured+vector | - | p50 0.061 / p95 0.113 |
| structured+rerank | p50 0.984 / p95 1.114 | p50 0.064 / p95 0.119 |

## Cost

| config | total USD | USD / question |
|---|---|---|
| naive+vector | 2.19 | 0.0429 |
| naive+rerank | 2.21 | 0.0433 |
| structured+vector | 2.32 | 0.0455 |
| structured+rerank | 2.28 | 0.0446 |

Generation and judging ran through the Message Batches API (50% price); generation latency is not measured in batch runs.
