module TestFactorizationMachines

using Base.Test

info("Testing - unknow learning method should throw an exception")
X = sparse(randn(10, 5))
y = randn(5)
@test_throws(Exception, fmTrain(X,y,:unknowMethod))

end
