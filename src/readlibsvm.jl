function read_libsvm(fname::ASCIIString, dimension = :col)
    label = Float64[]
    mI = Int64[]
    mJ = Int64[]
    mV = Float64[]
    fi = open(fname, "r")
    cnt = 1
    for line in eachline(fi)
        line = split(strip(line), " ")
        push!(label, parse(Float64, line[1]))
        line = line[2:end]
        for itm in line
            itm = split(itm, ":")
            push!(mI, parse(Int, itm[1]) + 1)
            push!(mJ, cnt)
            push!(mV, parse(Float64, itm[2]))
        end
        cnt += 1
    end
    close(fi)

    if dimension == :col 
        (sparse(mI,mJ,mV), label)
    else
        (sparse(mJ,mI,mV), label)
    end
end
