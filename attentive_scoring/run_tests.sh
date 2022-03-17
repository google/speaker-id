#!/bin/bash
set -o errexit

# Get project path.
PROJECT_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Add project modules to PYTHONPATH.
if [[ "${PYTHONPATH}" != *"${PROJECT_PATH}"* ]]; then
    export PYTHONPATH="${PYTHONPATH}:${PROJECT_PATH}"
fi

pushd ${PROJECT_PATH}

# Run tests.
python3 attentive_scoring_layer_test.py
echo "All tests passed!"

popd
