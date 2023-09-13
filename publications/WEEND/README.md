# Word-Level End-to-End Neural Speaker Diarization (WEEND)

## Overview

This repository contains supplementary details and information to the "Towards Word-Level End-to-End Neural Speaker Diarization with Auxiliary Network" paper.

## WDER and modified WDER without overlapping words

The Word Diarization Error Rate (WDER) reported in the paper follows "Joint Speech Recognition and Speaker Diarization via Sequence Transduction": https://arxiv.org/pdf/1907.05337.pdf. The metric is a word-level error metric without taking time boundaries into consideration, and it suits the word-level end-to-end diarization problem better.

For AMI in the paper, we evaluate and report a modified WDER which does not count words that overlap with any other word in the ground truth. The pseudocode looks like this:

```
TBA.
```

## Full evaluation results

We include the full evaluation results including both the WDER and the modified WDER on AMI.

**Table 2**

| Testsets         	| WDER Baseline 	| WDER Proposed 	|
|:----------------:	|:-------------:	|:-------------:	|
| Callhome         	| 10.3          	| 7.7           	|
| Fisher           	| 3.6           	| 8.0           	|
| AMI (modified)   	| 8.7           	| 50.0          	|
| AMI (unmodified) 	| 11.8          	| 52.3          	|
| Sim 2spk         	| 4.2           	| 4.1           	|
| Sim 3spk         	| 4.2           	| 3.6           	|
| Sim 4spk         	| 4.5           	| 5.1           	|

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

