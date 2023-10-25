# Word-Level End-to-End Neural Speaker Diarization (WEEND)

## Overview

This repository contains supplementary details and information to the "Towards Word-Level End-to-End Neural Speaker Diarization with Auxiliary Network" paper.

## Baseline with inserted speaker tags

We followed Shafey et al. [[1]](#1) and trained an RNN-T ASR model with augmented, inserted speaker tags. The training data was like this, with speaker labels re-formatted in an order-based manner **spk:N** :
> How are you doing today **spk:1** I am good **spk:2** Great **spk:1**

We reserved a few tokens from our 4096-sized WPM and used those to represent speakers 1-8:

```
(r'spk:1', '<addressnum>'),
(r'spk:2', '</addressnum>'),
(r'spk:3', '<app>'),
(r'spk:4', '</app>'),
(r'spk:5', '<apt>'),
(r'spk:6', '</apt>'),
(r'spk:7', '<areacode>'),
(r'spk:8', '</areacode>'),
```

For evaluation, we converted those special tokens back to speaker labels and evaluated them the same way as the other models in the paper. Full results of this model, denoted as **BaselineTag**, are included in [the full evaluation results section](#full-evaluation-results).

The results show that for generic speaker diarization task (instead of fixed role speaker classification problem), simply using inserted speaker tags does not work well. There is a slight degradation of WER on public datasets but small improvement on simulated LibriSpeech datasets. This is because the baseline RNN-T ASR model was not pre-trained on LibriSpeech, but in our ASR-diarization training, there were a large number of simulated utterances from LibriSpeech. Meanwhile, WDER is very high across the board. Even on the easiest 2-speaker simulated testset, the model has over 10% WDER.

## WDER and modified WDER without overlapping words

The Word Diarization Error Rate (WDER) reported in the paper follows "Joint Speech Recognition and Speaker Diarization via Sequence Transduction": https://arxiv.org/pdf/1907.05337.pdf. The metric is a word-level error metric without taking time boundaries into consideration, and it suits the word-level end-to-end diarization problem better.

For AMI in the paper, we evaluate and report a modified WDER which does not count words that overlap with any other word in the ground truth. Note that for AMI, *per-word* RTTM is avaiable.

The pseudocode looks like this:

```Python
''' Pseudo-code in Python for the modified WDER algorithm w/ ground truth metadata RTTM. '''

# Align ref and hyp text. align() returns alignment tuples
# of (i, j) indicating ith ref aligns with jth hyp. -1 values
# indicate deletions or insertions.
alignment = []
for i, j in align(ref, hyp):
  # Skip deletion or insertion.
  if i == -1 or j == -1:
    continue
  alignment.append((i, j))

# Assume annotations are sorted by start time.
overlap_idx = set()
prev_end = 0.0
for i, rttm in enumerate(rttms):
  if rttm.start < prev_end:
    overlap_idx.add(i - 1)
    overlap_idx.add(i)
  prev_end = max(rttm.end, prev_end)

updated_alignment = []
for i, j in alignment:
  # Save the ones that are not in the overlaps.
  if i not in overlap_idx:
    updated_alignment.append((i, j))
alignment = updated_alignment

# Use the updated alignment to find permutation invariant
# distance, e.g. via Hungarian algorithm.
...
```

## Full evaluation results

We include the full evaluation results including both the WDER and the modified WDER on AMI.

**Table 2 WER (S/D/I)**

| Testsets         	| WER BaselineTag 	| WER Baseline 		| WER Proposed 		|
|:----------------:	|:-------------:	|:-------------:	|:-------------:	|
| Callhome         	| 47.2 (14.3/5.3/27.6)	| 45.9 (12.8/9.7/23.3)  | 45.9 (12.8/9.7/23.3)  |
| Fisher           	| 25.1 (8.4/14.0/2.7)	| 20.5 (8.7/10.4/1.4)   | 20.5 (8.7/10.4/1.4)   |
| AMI		   	| 77.6 (3.3/73.8/0.5)	| 29.6 (8.9/19.9/0.8)	| 29.6 (8.9/19.9/0.8)	|
| Sim 2spk         	| 6.4 (5.0/0.7/0.7)	| 8.1 (6.4/1.0/0.7)	| 8.1 (6.4/1.0/0.7)	|
| Sim 3spk         	| 6.3 (5.1/0.6/0.7) 	| 8.3 (6.5/1.0/0.8)	| 8.3 (6.5/1.0/0.8)	|
| Sim 4spk         	| 6.4 (5.1/0.6/0.7)    	| 8.1 (6.4/1.0/0.7)	| 8.1 (6.4/1.0/0.7)	|

Note: **WER BaselineTag** deletion error on AMI is super high. This is because without too much tuning, our trained model has a relatively high probability to give up decoding and return empty hypothesis text if the audio gets extremely long. This issue is only observed on AMI testset.

<br />

**Table 2 WDER**

| Testsets         	| WDER BaselineTag 	| WDER Baseline 	| WDER Proposed 	|
|:----------------:	|:-------------:	|:-------------:	|:-------------:	|
| Callhome         	| 39.4          	| 10.3          	| 7.7           	|
| Fisher           	| 41.5           	| 3.6           	| 8.0           	|
| AMI (modified)   	| 62.8           	| 8.7           	| 50.0          	|
| AMI (unmodified) 	| 61.1          	| 11.8          	| 52.3          	|
| Sim 2spk         	| 11.9           	| 4.2           	| 4.1           	|
| Sim 3spk         	| 16.7           	| 4.2           	| 3.6           	|
| Sim 4spk         	| 27.0           	| 4.5           	| 5.1           	|

<br />

**Table 3**

| Testsets         	| Short-form Lengths (s) 	| WDER Baseline (%) 	| WDER Proposed (%) 	|
|:----------------:	|:----------------------:	|:-----------------:	|:-----------------:	|
| Callhome         	| 30                     	| 13.6              	| 9.3               	|
|                  	| 60                     	| 9.8               	| 8.9               	|
|                  	| 120                    	| 10.5              	| 8.9               	|
| Fisher           	| 30                     	| 8.6               	| 3.8               	|
|                  	| 60                     	| 4.8               	| 3.7               	|
|                  	| 120                    	| 4.0               	| 3.7               	|
| AMI (modified)   	| 30                     	| 10.1              	| 9.9               	|
|                  	| 60                     	| 6.7               	| 13.3              	|
|                  	| 120                    	| 8.0               	| 18.8              	|
| AMI (unmodified) 	| 30                     	| 13.0              	| 16.0              	|
|                  	| 60                     	| 10.1              	| 19.3              	|
|                  	| 120                    	| 11.4              	| 24.4              	|

<br />

**Table 4**

AMI modified:

| AMI<br>Lengths 	| Baseline<br>WDER 	|      	|      	|      	| Proposed<br>WDER 	|      	|      	|      	|
|:--------------:	|:----------------:	|:----:	|:----:	|:----:	|:----------------:	|:----:	|:----:	|:----:	|
|                	| 1spk             	| 2spk 	| 3spk 	| 4spk 	| 1spk             	| 2spk 	| 3spk 	| 4spk 	|
| 30sec          	| 18.6             	| 10.0 	| 8.8  	| 8.4  	| 1.1              	| 5.8  	| 10.1 	| 15.5 	|
| 60sec          	| 10.8             	| 6.3  	| 5.6  	| 6.9  	| 0.8              	| 5.2  	| 12.1 	| 17.1 	|
| 120sec         	| -                	| 6.4  	| 4.4  	| 9.3  	| -                	| 9.8  	| 15.8 	| 20.8 	|

<br/>

AMI unmodified:

| AMI<br>Lengths 	| Baseline<br>WDER 	|      	|      	|      	| Proposed<br>WDER 	|      	|      	|      	|
|:--------------:	|:----------------:	|:----:	|:----:	|:----:	|:----------------:	|:----:	|:----:	|:----:	|
|                	| 1spk             	| 2spk 	| 3spk 	| 4spk 	| 1spk             	| 2spk 	| 3spk 	| 4spk 	|
| 30sec          	| 18.5             	| 11.0 	| 11.9 	| 13.4 	| 1.1              	| 8.2  	| 16.1 	| 23.8 	|
| 60sec          	| 10.8             	| 7.1  	| 8.6  	| 11.3 	| 0.8              	| 6.7  	| 17.5 	| 24.1 	|
| 120sec         	| -                	| 6.9  	| 6.9  	| 13.2 	| -                	| 11.7 	| 20.4 	| 26.8 	|

<br />

**Table 5**

| Intermediate<br>Layer Selection 	| Callhome 	| Fisher 	| Simulated 	| AMI Short<br>(modified) 	| AMI Short<br>(unmodified) 	|
|---------------------------------	|:--------:	|:------:	|:---------:	|:-----------------------:	|:-------------------------:	|
| 0th Conf layer (features)       	| 23.8     	| 24.5   	| 10.4      	| 22.8                    	| 28.1                      	|
| 5th Conf layer (proposed)       	| 7.7      	| 8.0    	| 4.3       	| 14.0                    	| 19.9                      	|
| 12th Conf layer (last)          	| 33.6     	| 37.3   	| 46.9      	| 27.5                    	| 32.5                      	|

<br />

**Table 6**

| Model            	        | Callhome 	| Callhome Short 	| Fisher 	| Fisher Short 	| Simulated 	| AMI Short<br>(modified) 	| AMI Short<br>(unmodified) 	|
|-------------------------- |:--------:	|:--------------:	|:------:	|:------------:	|:---------:	|:-----------------------:	|:-------------------------:	|
| Proposed                 	| 7.7      	| 9.0            	| 8.0    	| 3.7          	| 4.3       	| 14.0                    	| 19.9                      	|
| &nbsp;-simulated          | 11.6     	| 9.8            	| 12.3   	| 5.4          	| 22.2      	| 19.0                    	| 24.8                      	|
| &nbsp;&nbsp;-30/60s segs 	| 28.8     	| 22.5           	| 22.1   	| 15.7         	| 26.8      	| 26.2                    	| 32.1                      	|


## References
<a id="1">[1]</a> Laurent El Shafey et al., “Joint speech recognition and speaker diarization via sequence transduction,” in Interspeech, 2019, pp. 396–400.
