module SimpleRecoTest

using FactorizationMachines
using Base.Test

# simple recommendation example

 #1 is User1
 #2 is User2
 #3 is Item1,
 #4 is Item2,
 #5 is Item3,
 #6 is Item4
 #7 is the category “18-25″,
 #8 is the category “26-40″,
 #9 is the category “40-60″
 #10 will represent the price feature

#rating     #user   #items     #category    #price

T = [
5           1 0     1 0 0 0    1 0 0        12.5; 
5           1 0     0 1 0 0    1 0 0        20; 
4           1 0     0 0 1 0    1 0 0        78;
1           0 1     1 0 0 0    0 0 1        12.5;
1           0 1     0 1 0 0    0 0 1        20;
]



info("Testing reading from libsvm format...")

(XFromFile, yFromFile) = fmReadLibSVM("data/small_train.libfm")

println(XFromFile)
println(size(XFromFile))
@test size(XFromFile) == (10, 5)

X = sparse(T[:,2:end])
y = T[:,1]

fm = fmTrain(sparse(XFromFile),yFromFile)

(TFromFile, tFromFile) = fmReadLibSVM("data/small_test.libfm")
p = fmPredict(fm,sparse(TFromFile))

info("Predictions: $p")

end
