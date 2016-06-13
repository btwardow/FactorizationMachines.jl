#!/usr/bin/env bash

echo "Running fastFM benchmark"
time python fastfm_py_benchmark.py
echo ""

echo "Running FM Julia benchmark"
time julia fm_jl_benchmark.jl
