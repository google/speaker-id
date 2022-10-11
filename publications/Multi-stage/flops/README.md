# FLOPs analysis

## Introduction

Here is the script to estimate the total number of floating point operations (FLOPs) required to perform one individual clustering step for multi-stage clustering.

`estimate_flops.py` is the script, and `output.csv` is the result.

## Dependencies

To run the script, you need to install these dependencies:

```
pip install spectralcluster
pip install python_papi
pip install scikit-learn
```

## Using pypapi

On a Linux machine, to make sure pypapi has the permission to collect stats, you need to run this command every time before you run the script:

```
sudo sh -c 'echo 1 >/proc/sys/kernel/perf_event_paranoid'
```

Also, on some machines or operating systems, you may not be able to collect the `PAPI_FP_OPS` events. You can run the command below to check what events are available:

```
sudo apt install papi-tools

papi_avail
```