module Common
export sqerr,
       sqerr_deriv,
       nlogsig,
       nlogsig_deriv,
       heaviside,
       sigmoid

"""Squared error"""
sqerr(p::Number, y::Number)       = (p - y)^2
sqerr_deriv(p::Number, y::Number) = 2(p - y)

"""Negative logistic sigmoid"""
nlogsig(p::Number, y::Number)       = -log(sigmoid(p * y))
nlogsig_deriv(p::Number, y::Number) = y * sigmoid(p * y) - y

"""Heaviside step function"""
heaviside(p::Number, y::Number) = p == y ? zero(p) : one(p)

"""Sigmoid"""
sigmoid(x::Number) = one(x) / (one(x) + exp(-x))

end
