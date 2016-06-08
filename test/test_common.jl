module TestFactorizationMachines

using FactorizationMachines
using Base.Test

info("Testing squared error loss")
y =  [1.0, 1.0, -1.0, -1.0, 0.5]
yhat = [1.0, -1.0, 1.0, -1.0, -0.25]
expected_loss = [0, 4, 4, 0, 0.5625]
for i in 1:length(y)
  @test expected_loss[i] == FactorizationMachines.sqerr(yhat[i], y[i])
end

info("Testing 0-1 loss")
y =  [1.0, 1.0, -1.0, -1.0]
yhat = [1.0, -1.0, 1.0, -1.0]
expected_loss = [0, 1, 1, 0]
for i in 1:length(y)
  @test expected_loss[i] == FactorizationMachines.z1loss(yhat[i], y[i])
end

end
