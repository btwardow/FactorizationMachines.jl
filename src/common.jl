"""Squared error"""
sqerr(p::Number, y::Number) = (p - y)^2
sqerr_deriv(p::Number, y::Number) = (p - y)

"""Logistic loss"""
logloss(p::Number, y::Number) = -log(sigmoid(-p * y))
logloss_deriv(p::Number, y::Number) = y * (sigmoid(-p * y) - 1)

"""0-1 loss"""
z1loss(p::Number, y::Number) = p == y ? 0 : 1

"""Sigmoid"""
sigmoid(x::Number) = 1 / (1 + exp(-x))
function sigmoid_deriv(x::Number)
  s = sigmoid(x)
  s * (1 - s)
end
