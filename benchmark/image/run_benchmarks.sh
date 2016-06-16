#!/usr/bin/env bash

# Change to the directory of the current script
cd "$(dirname "$0")"

# download MovieLense datasets
TRAIN_FILENAME="data/ml-10m.train.libfm"
TEST_FILENAME="data/ml-10m.test.libfm"

if [ ! -f "$TRAIN_FILENAME" ] || [ ! -f "$TEST_FILENAME" ]
then 
    echo "Downloading MovieLens 1OM dataset (6M)..."
    curl -s -o ml-10m.zip http://files.grouplens.org/datasets/movielens/ml-10m.zip
    unzip ml-10m.zip
    rm ml-10m.zip

    echo "Preparing train and tst files in svmlight format..."
    mkdir data
    python prepare_data.py ml-10M100K/ratings.dat $TRAIN_FILENAME $TEST_FILENAME
    rm -rf ml-10M100K
fi

# simple check if fastFM package is available
python -c 'import fastFM' || echo 'Please install fastFM package: "pip install fastFM"'

echo "Running fastFM benchmark"
time python fastfm_py_benchmark.py $TRAIN_FILENAME $TEST_FILENAME
echo ""

echo "Running FM Julia benchmark"
time julia fm_jl_benchmark.jl $TRAIN_FILENAME $TEST_FILENAME

