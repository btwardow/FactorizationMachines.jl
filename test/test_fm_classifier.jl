module TestFactorizationMachines

using FactorizationMachines
using Base.Test

info("Testing classifier 0-1 loss")
y =  [1.0, 1.0, -1.0, -1.0]
yhat = [1.0, -1.0, 1.0, -1.0]
expected_loss = [0, 1, 1, 0]
for i in 1:length(y)
  @test expected_loss[i] == FactorizationMachines.fmLoss(FactorizationMachines.FMClassifier, yhat[i], y[i])
end

end
