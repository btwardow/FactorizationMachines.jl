module TestFMDatasetML100K

using FactorizationMachines
using Base.Test

info("Testing reading from libsvm format...")

# read data in columnar sparse matrix
XFromFile, yFromFile = read_libsvm("data/small_train.libfm")

@test typeof(XFromFile) == SparseMatrixCSC{Float64,Int64}
I, J, V = findnz(XFromFile)

# check observations
@test size(XFromFile) == (10, 5)
@test I == [1,3,7,10,1,4,7,10,1,5,7,10,2,3,9,10,2,4,9,10]
@test J == [1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5]
@test V == [1.0,1.0,1.0,12.5,1.0,1.0,1.0,20.0,1.0,1.0,1.0,78.0,1.0,1.0,1.0,12.5,1.0,1.0,1.0,20.0]

#check predictions
@test typeof(yFromFile) == Array{Float64,1}
@test size(yFromFile, 1) == 5

info("Test reading from libsvm formate ended.")
end
