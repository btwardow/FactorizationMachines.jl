module Common
export sqerr,
       sqerr_deriv,
       nlogsig,
       nlogsig_deriv,
       heaviside,
       sigmoid,
       sigmoid_deriv

"""Squared error"""
sqerr(p::Number, y::Number)       = (p - y)^2
sqerr_deriv(p::Number, y::Number) = (p - y)

"""Negative logistic sigmoid"""
nlogsig(p::Number, y::Number)       = -log(sigmoid(-p * y))
nlogsig_deriv(p::Number, y::Number) = y * (sigmoid(-p * y) - 1)

"""Heaviside step function"""
heaviside(p::Number, y::Number) = p == y ? 0 : 1

"""Sigmoid"""
sigmoid(x::Number) = 1 / (1 + exp(-x))
function sigmoid_deriv(x::Number)
    s = sigmoid(x)
    s * (1 - s)
end

end
