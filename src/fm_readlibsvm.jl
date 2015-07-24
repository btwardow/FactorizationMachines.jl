function fmReadLibSVM(fname::ASCIIString, dimension = :col)
    label = Float64[]
    mI = Int64[]
    mJ = Int64[]
    mV = Float64[]
    fi = open(fname, "r")
    cnt = 1
    for line in eachline(fi)
        line = split(strip(line), " ")
        push!(label, float(line[1]))
        line = line[2:end]
        for itm in line
            itm = split(itm, ":")
            push!(mI, int(itm[1]) + 1)
            push!(mJ, cnt)
            push!(mV, float(itm[2]))
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
