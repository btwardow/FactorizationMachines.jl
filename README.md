# FactorizationMachines.jl

[![Build Status](https://travis-ci.org/btwardow/FactorizationMachines.jl.svg?branch=master)](https://travis-ci.org/btwardow/FactorizationMachines.jl)[![Coverage Status](https://coveralls.io/repos/btwardow/FactorizationMachines.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/btwardow/FactorizationMachines.jl?branch=master)

Implementation of Factorization Machines for Julia. 

Types of Implemented FM:
 - Standard (libfm.org) with SGD and only for regression
 - (TODO) Field-Aware (http://www.csie.ntu.edu.tw/~r01922136/libffm)
 - (TODO) Gaussian Process (http://www.ci.tuwien.ac.at/~alexis/Publications_files/gpfm-sigir14-draft.pdf)


# TODOs:
 - performance changes for Julia
 - classifiation 
 - adaptive SGD
 - suffle training set
 - performance benchmark
