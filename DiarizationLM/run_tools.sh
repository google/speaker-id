#!/bin/bash
set -o errexit
set -o nounset
set -o pipefail
set -o xtrace

python3 train_data_prep.py \
--input=testdata/example_data.json \
--output=/tmp/example_data.tfrecord \
--output_type=tfrecord

python3 postprocess_completions.py \
--input=testdata/example_completion_with_bad_completion.json \
--output=/tmp/example_postprocessed.json
