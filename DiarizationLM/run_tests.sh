#!/bin/bash
set -o errexit
set -o nounset
set -o pipefail
set -o xtrace

python3 -m flake8 --indent-size 2 --max-line-length 80 .
python3 -m pytype .
python3 levenshtein_test.py
python3 utils_test.py

python3 train_data_prep.py \
--input=testdata/example_data.json \
--output=/tmp/example_data.tfrecord \
--output_type=tfrecord

python3 postprocess_completions.py \
--input=testdata/example_completion_with_bad_completion.json \
--output=/tmp/example_postprocessed.json
