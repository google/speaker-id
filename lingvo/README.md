# Lingvo-based modules for speaker and language recognition

## Overview

Here we open source some of the [Lingvo](https://github.com/tensorflow/lingvo)-based
modules used in our publications.

## Disclaimer

**This is NOT an official Google product.**

## GE2E and GE2E-XS losses

GE2E and GE2E-XS losses are implemented in `lingvo/loss_layers.py`.

GE2E was proposed in this paper:

* [Generalized End-to-End Loss for Speaker Verification](https://arxiv.org/abs/1710.10467)

GE2E-XS was proposed in this paper:

* [Dr-Vectors: Decision Residual Networks and an Improved Loss for Speaker Recognition](https://arxiv.org/abs/2104.01989)

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
@inproceedings{wan2018generalized,
  title={Generalized end-to-end loss for speaker verification},
  author={Wan, Li and Wang, Quan and Papir, Alan and Moreno, Ignacio Lopez},
  booktitle={International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={4879--4883},
  year={2018},
  organization={IEEE}
}

@inproceedings{pelecanos2021drvectors,
  title={{Dr-Vectors: Decision Residual Networks and an Improved Loss for Speaker Recognition}},
  author={Jason Pelecanos and Quan Wang and Ignacio Lopez Moreno},
  year={2021},
  booktitle={Proc. Interspeech},
  pages={4603--4607},
  doi={10.21437/Interspeech.2021-641}
}

@inproceedings{pelecanos2022parameter,
  title={Parameter-Free Attentive Scoring for Speaker Verification},
  author={Jason Pelecanos and Quan Wang and Yiling Huang and Ignacio Lopez Moreno},
  booktitle={Odyssey: The Speaker and Language Recognition Workshop},
  year={2022}
}

@inproceedings{wang2022attentive,
  title={Attentive Temporal Pooling for Conformer-based Streaming Language Identification in Long-form Speech},
  author={Quan Wang and Yang Yu and Jason Pelecanos and Yiling Huang and Ignacio Lopez Moreno},
  booktitle={Odyssey: The Speaker and Language Recognition Workshop},
  year={2022}
}
```
