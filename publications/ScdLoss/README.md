# Augmenting Transformer-Transducer Based Speaker Change Detection With Token-Level Training Loss

## Introduction

This repository hosts supplemental results and resources for the manuscript
titled "[Augmenting Transformer-Transducer Based Speaker Change Detection With
Token-Level Training Loss](https://arxiv.org/abs/2211.06482)".

## Significance tests

Some additional information may help the reader better interpret the findings in
the manuscript. Therefore, we conduct utterance-level paired two-sample
two-sided t-tests between the three systems (Baseline, EMBR, and SCD loss) on
the mean value of the scores using the `scipy.stats.ttest_rel`
[library](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_rel.html).
We treat each utterance as a test unit, and compute the various scores produced
by different systems as samples (observations). The null hypothesis is that "the
two samples of scores have identical average (expected) scores."

For the long-form test data (N=1,025), only the following t-tests have a
p-value > 0.05,

*   EMBR vs. Baseline: the coverage score (p=0.27).
*   SCD loss vs. Baseline: the precision score defined in section 3.1 (p=0.37).
*   SCD loss vs. EMBR: the F1 score of purity and coverage (p=0.08).

For the short-form test data, on 30s segments (N=14,458), the t-test on the
average coverage score between the SCD loss and EMBR systems has a p-value of
0.33.

All other t-tests have a p-value << 0.01, meaning that we reject the null
hypothesis of equal averages.

## Additional `pyannote.metrics` results

In the table below, please find the full results that correspond to the inline `pyannote.metrics`
precision and recall results described in section 5.1.

We present precision, recall, and F1 rates following `pyannote.metrics`
[definitions](https://pyannote.github.io/pyannote-metrics/reference.html#segmentation)
described in the first paragraph of section 3. We use a collar value of 250ms on
each side of the speaker change point.

| Metric        | System   | AMI      | CallHome | DIHARD1  | Fisher   | ICSI     | Inbound  | Outbound | Pooled data |
| :-----------: | :------: | :------: | :------: | :------: | :------: | :------: | :------: | :------: | :---------: |
|               | Baseline | 25.6     | 43.7     | 30.6     | 35.4     | **33.6** | 22.2     | 33.3     | 31.5        |
| Precision (\%)| EMBR     | **26.1** | 44.8     | 31.5     | 36.3     | **33.6** | 25.6     | **35.3** | 33.1        |
|               | SCD loss | 25.3     | **46.7** | **33.1** | **36.5** | 33.4     | **27.3** | 34.9     | **33.3**    |
|               |          |          |          |          |          |          |          |          |             |
|               | Baseline | 36.3     | 33.5     | 26.1     | 39.2     | 26.9     | 25.0     | 34.1     | 33.0        |
| Recall (\%)   | EMBR     | 37.4     | 35.6     | 26.7     | 43.8     | 26.5     | 31.8     | 37.1     | 36.3        |
|               | SCD loss | **40.2** | **40.8** | **29.9** | **47.1** | **29.4** | **38.0** | **41.4** | **40.2**    |
|               |          |          |          |          |          |          |          |          |             |
|               | Baseline | 30.0     | 38.0     | 28.2     | 37.2     | 29.9     | 23.5     | 33.7     | 32.2        |
| F1 (\%)       | EMBR     | 30.7     | 39.7     | 28.9     | 39.7     | 29.6     | 28.3     | 36.2     | 34.6        |
|               | SCD loss | **31.1** | **43.5** | **31.4** | **41.1** | **31.3** | **31.7** | **37.9** | **36.5**    |
