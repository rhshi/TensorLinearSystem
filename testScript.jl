include("testing.jl")
using Printf, JLD2


println("Enter a filename:")
flush(stdout)
fName_ = readline()
fName = "results/$fName_.jld2"

print("\n")
flush(stdout)

println("Enter first n:")
flush(stdout)
nStart_ = readline()
nStart = parse(Int64, nStart_)

print("\n")
flush(stdout)

println("Enter last n:")
flush(stdout)
nEnd_ = readline()
nEnd = parse(Int64, nEnd_)

jldopen(fName, "w") do file
    file["nStart"] = nStart
    file["nEnd"] = nEnd
end;

print("\n")
flush(stdout)

println("Gathering statisics for n=$nStart,...,$nEnd\n")

for n=nStart:nEnd
    println("n=$n")
    flush(stdout) 
    c = 0
    for r=n+2:binomial(n+2, 2)
        A = randn(n+1, r)/sqrt(n+1) + im * randn(n+1, r)/sqrt(n+1)
        A2_ = khatri_rao(A, 2);
        T = reshape(A2_*permutedims(A2_), (n+1, n+1, n+1, n+1));
        basis_inds = collect(1:r);
        
        X, Y = linearSystemTesting(T, basis_inds);
     
        if isnothing(X)
            break
        else
            foreach(normalize!, eachrow(X))
            foreach(normalize!, eachrow(Y))
            
            c += 1
            
            eqNum, varNum = size(X)
#             singValsX = svdvals(X)
#             singValsY = svdvals(Y)
            
            rValsX = sort(abs.(diag(qr(X).R)), rev=true)
            rValsY = sort(abs.(diag(qr(Y).R)), rev=true)
            
            jldopen(fName, "a+") do file
                file["$n/$r/numVars"] = varNum
                file["$n/$r/numEqs"] = eqNum
                
#                 file["$n/$r/singVals"] = singValsX
#                 file["$n/$r/rank"] = processRank(singValsX)
#                 file["$n/$r/singValsGenericCoeffs"] = singValsY
#                 file["$n/$r/rankGenericCoeffs"] = processRank(singValsY)
                
                file["$n/$r/rVals"] = rValsX
                file["$n/$r/rank"] = processRank(rValsX)
                file["$n/$r/rValsGenericCoeffs"] = rValsY
                file["$n/$r/rankGenericCoeffs"] = processRank(rValsY)
                
            end
        end
    end
    jldopen(fName, "a+") do file
        file["$n/numR"] = c
    end
end