module TestFactorizationMachines

using FactorizationMachines
using Base.Test

info("Testing squared error loss")
y =  [1.0, 1.0, -1.0, -1.0, 0.5]
yhat = [1.0, -1.0, 1.0, -1.0, -0.25]
expected_loss = [0, 4, 4, 0, 0.5625]
for i in 1:length(y)
  @test expected_loss[i] == Common.sqerr(yhat[i], y[i])
end

info("Testing 0-1 loss")
y =  [1.0, 1.0, -1.0, -1.0]
yhat = [1.0, -1.0, 1.0, -1.0]
expected_loss = [0, 1, 1, 0]
for i in 1:length(y)
  @test expected_loss[i] == Common.heaviside(yhat[i], y[i])
end

info("Testing negative logistic sigmoid loss for {1, -1} labels")
y =  [1.0, 1.0, -1.0, -1.0]
yhat = [1.0, -1.0, 1.0, -1.0]
expected_loss = [-log(Common.sigmoid(1)), log(1 + e), log(1 + e), -log(Common.sigmoid(1))]
for i in 1:length(y)
  @test expected_loss[i] == Common.nlogsig(yhat[i], y[i])
end

info("Testing negative logistic sigmoid loss for {1, 0} labels")
y =  [1.0, 1.0, 0.0, 0.0]
yhat = [1.0, 0.0, 1.0, 0.0]
expected_loss = [-log(Common.sigmoid(1)), log(2), log(2), log(2)]
for i in 1:length(y)
  @test expected_loss[i] == Common.nlogsig(yhat[i], y[i])
end

end
