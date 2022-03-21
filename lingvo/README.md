# Lingvo-based modules for speaker and language recognition

## Overview

Here we open source some of the [Lingvo](https://github.com/tensorflow/lingvo)-based
modules used in our publications.

## Disclaimer

**This is NOT an official Google product.**

## Attentive temporal pooling

Attentive temporal pooling is implemented in `lingvo/cumulative_statistics_layer.py`.

It is used by these papers:

* [Attentive Temporal Pooling for Conformer-based Streaming Language Identification in Long-form Speech](https://arxiv.org/abs/2202.12163)
* [Parameter-Free Attentive Scoring for Speaker Verification](https://arxiv.org/abs/2203.05642)

## Attentive scoring

Attentive scoring is implemented in `lingvo/attentive_scoring_layer.py`.

It is proposed in this paper:

* [Parameter-Free Attentive Scoring for Speaker Verification](https://arxiv.org/abs/2203.05642)


## Citations

Our papers are cited as:

```
@article{pelecanos2022parameter,
  title={Parameter-Free Attentive Scoring for Speaker Verification},
  author={Jason Pelecanos and Quan Wang and Yiling Huang and Ignacio Lopez Moreno},
  journal={arXiv preprint arXiv:2203.05642},
  year={2022}
}

@article{wang2022attentive,
  title={Attentive Temporal Pooling for Conformer-based Streaming Language Identification in Long-form Speech},
  author={Quan Wang and Yang Yu and Jason Pelecanos and Yiling Huang and Ignacio Lopez Moreno},
  journal={arXiv preprint arXiv:2202.12163},
  year={2022}
}
```
