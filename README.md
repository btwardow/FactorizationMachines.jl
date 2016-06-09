# FactorizationMachines.jl

[![Build Status](https://travis-ci.org/btwardow/FactorizationMachines.jl.svg?branch=master)](https://travis-ci.org/btwardow/FactorizationMachines.jl)[![Coverage
Status](https://coveralls.io/repos/btwardow/FactorizationMachines.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/btwardow/FactorizationMachines.jl?
branch=master)

## Implementation of Factorization Machines for Julia.

As author describes:

> FMs combine the high-prediction accuracy of factorization models with the flexibility of feature engineering.
> The input data for FMs is described with real-valued features, exactly like in other machine-learning
> approaches such as linear regression, support vector machines, etc.
> However, the internal model of FMs uses factorized interactions between variables,
> and thus, it shares with other factorization models the high prediction quality in sparse settings,
> like in recommender systems. It has been shown that FMs can mimic most factorization
> models just by feature engineering [Rendle 2010]

Steffen Rendle (2012): Factorization Machines with libFM, in ACM Trans. Intell. Syst. Technol., 3(3), May. [PDF]

Implementation is mostly based on [libfm software](libfm.org) and try to be compatible with conventions used there.

## Usage

### Simple recommendation example

```julia
using FactorizationMachines

T = [
5           1 0     1 0 0 0    1 0 0        12.5;
5           1 0     0 1 0 0    1 0 0        20;
4           1 0     0 0 1 0    1 0 0        78;
1           0 1     1 0 0 0    0 0 1        12.5;
1           0 1     0 1 0 0    0 0 1        20;
]

X = sparse(T[:,2:end])'
y = T[:,1]

fm = train(X, y)

X_new = sparse([
0 1     0 1 0 0    0 0 1        13.0;
])'

p = predict(fm, X_new)
```

### Using LIBSVM file format

```julia
(X_train, y_train) = read_libsvm("data/small_train.libfm")
fm = train(sparse(X_train), y_train)
(X_test, y_test) = read_libsvm("data/small_test.libfm")
p = predict(fm, sparse(X_test))
```


## TODOs:
-   AdaGrad SGD optimization
-   Performance benchmark with libfm and python implementation (pyfm/fastFM/lightfm)
-   [Field-Aware FM](http://www.csie.ntu.edu.tw/~r01922136/libffm)
-   [Gaussian Process FM](http://www.ci.tuwien.ac.at/~alexis/Publications_files/gpfm-sigir14-draft.pdf)
-   MCMC and ALS - just like in libfm
