include("util.jl") 


function processRank(singVals; tol=1e-10)
    for k=1:length(singVals)-1
        sVal1 = singVals[k]
        sVal2 = singVals[k+1]
        
        if sVal2/sVal1 <= tol
           return k 
        end
        
    end
    return length(singVals)
end;

function linearSystem(T, basis_inds)
    n = size(T)[1]-1
    d = length(size(T))
    
    D, Drev = makeDicts(n+1, d);

    basis_inds = collect(1:r)
    basis, basisD = basisFn(basis_inds, Drev);

    gamma = Int(floor(d/2))
    delta = Int(floor(d/2))-1;

    alphas = basis[basisD[gamma]]
    betas = basis[basisD[delta]];
    
    TcatGam = catMat(T, gamma)
    TcatDelt = catMat(T, delta)
    H0 = TcatGam[basis_inds, basis_inds];

    H0_adj = cofactor(H0);
    H0_det = det(H0);

    varTups = varsInds(alphas, D);
    varTups = sort([var for var in varTups], rev=true)
    
    eqTups1, eqTups2 = linEqsTups(alphas, betas, n+1, D, basis, basis_inds);
    for eqTupPair in eqTups2
        if (eqTupPair[1] in eqTups1) && (eqTupPair[2] in eqTups1)
            delete!(eqTups2, eqTupPair)
        end
    end
    
    eqTups1 = sort([eqTup for eqTup in eqTups1], rev=true)
    eqTups2 = sort([eqTup for eqTup in eqTups2], rev=true);
    
    coeffDict = Dict()
    for (i, varTup) in enumerate(varTups)
       coeffDict[varTup] = i
    end
    
    X1 = Array{Complex, 2}(undef, (length(eqTups1), length(varTups)))
    y1 = zeros(Complex, length(eqTups1));

    X2 = Array{Complex, 2}(undef, (length(eqTups2), length(varTups)))
    y2 = zeros(Complex, length(eqTups2));
    
    for (k, eqTup) in enumerate(eqTups1)
        coeffRow, constant = processLinEq(eqTup, D, coeffDict, alphas, basis_inds, H0_det, H0_adj, TcatGam, TcatDelt)
        X1[k, :] = coeffRow
        y1[k] = constant
    end
    
    for (k, eqTupPair) in enumerate(eqTups2)
        eqTupL = eqTupPair[1]
        eqTupR = eqTupPair[2]

        coeffRowL, constantL = processLinEq(eqTupL, D, coeffDict, alphas, basis_inds, H0_det, H0_adj, TcatGam, TcatDelt)
        coeffRowR, constantR = processLinEq(eqTupR, D, coeffDict, alphas, basis_inds, H0_det, H0_adj, TcatGam, TcatDelt)

        X2[k, :] = coeffRowL - coeffRowR
        y2[k] = constantL - constantR
    end
    
    X1 ./= sqrt(Complex.(det(H0)))
    y1 ./= sqrt(Complex.(det(H0)))

    X2 ./= sqrt(Complex.(det(H0)))
    y2 ./= sqrt(Complex.(det(H0)));
    
    return X1, y1, X2, y2, varTups, eqTups1, eqTups2
end;

function linearSystemTesting(T, basis_inds)
    n = size(T)[1]-1
    d = length(size(T))
    
    D, Drev = makeDicts(n+1, d);

    basis, basisD = basisFn(basis_inds, Drev, d);

    gamma = Int(floor(d/2))
    delta = Int(floor(d/2))-1;

    alphas = basis[basisD[gamma]]
    betas = basis[basisD[delta]];
    
    TcatGam = catMat(T, gamma)
    TcatDelt = catMat(T, delta)
    H0 = TcatGam[basis_inds, basis_inds];

    H0_adj = cofactor(H0);
    H0_det = det(H0);

    varTups = varsInds(alphas, D);
    varTups = sort([var for var in varTups], rev=true)
    
    eqTups1, eqTups2 = linEqsTups(alphas, betas, n+1, D, basis, basis_inds);
    for eqTupPair in eqTups2
        if (eqTupPair[1] in eqTups1) && (eqTupPair[2] in eqTups1)
            delete!(eqTups2, eqTupPair)
        end
    end
    
    eqTups1 = sort([eqTup for eqTup in eqTups1], rev=true)
    eqTups2 = sort([eqTup for eqTup in eqTups2], rev=true);
    
    if length(eqTups1) + length(eqTups2) < length(varTups)
        return nothing, nothing
    else
    
        coeffDict = Dict()
        for (i, varTup) in enumerate(varTups)
           coeffDict[varTup] = i
        end
        
        ####################################
        
        b_js = Set(vcat(getfield.(eqTups1, 2), getfield.(getfield.(eqTups2, 1), 2), getfield.(getfield.(eqTups2, 2), 2)))
        genericDict = Dict()
        for alpha in alphas
            for b_j in b_js
                genericDict[(alpha[2:end], b_j)] = randn()
            end
        end
        detValue = randn()

        
        ####################################

        X1 = Array{Complex, 2}(undef, (length(eqTups1), length(varTups)))
        X2 = Array{Complex, 2}(undef, (length(eqTups2), length(varTups)))
        
        Y1 = Array{Float64, 2}(undef, (length(eqTups1), length(varTups)))
        Y2 = Array{Float64, 2}(undef, (length(eqTups2), length(varTups)))

        for (k, eqTup) in enumerate(eqTups1)
            coeffRow, _ = processLinEq(eqTup, D, coeffDict, alphas, basis_inds, H0_det, H0_adj, TcatGam, TcatDelt, d)
            X1[k, :] = coeffRow
            
            coeffRow2 = processLinEqGeneric(eqTup, D, coeffDict, genericDict, alphas, d, detValue)
            Y1[k, :] = coeffRow2
        end

        for (k, eqTupPair) in enumerate(eqTups2)
            eqTupL = eqTupPair[1]
            eqTupR = eqTupPair[2]

            coeffRowL, _ = processLinEq(eqTupL, D, coeffDict, alphas, basis_inds, H0_det, H0_adj, TcatGam, TcatDelt, d)
            coeffRowR, _ = processLinEq(eqTupR, D, coeffDict, alphas, basis_inds, H0_det, H0_adj, TcatGam, TcatDelt, d)

            X2[k, :] = coeffRowL - coeffRowR
            
            coeffRowL2 = processLinEqGeneric(eqTupL, D, coeffDict, genericDict, alphas, d, detValue)
            coeffRowR2 = processLinEqGeneric(eqTupR, D, coeffDict, genericDict, alphas, d, detValue)
            Y2[k, :] = coeffRowL2 - coeffRowR2
        end

        X1 ./= sqrt(Complex.(det(H0)))
        X2 ./= sqrt(Complex.(det(H0)))

        X = vcat(X1, X2)
        Y = vcat(Y1, Y2)

        return X, Y
        
    end
    
end;


function makeStats(f)
    
    function getPossR(n)
        c = f["$n/numR"]
        return n+2, n+1+c
    end;

    function getNumVars(n, r)
        return f["$n/$r/numVars"]
    end;

    function getNumEqs(n, r)
        return f["$n/$r/numEqs"]
    end;

    function getSingVals(n, r)
        return f["$n/$r/singVals"]
    end;

    function getRank(n, r)
        return f["$n/$r/rank"]
    end;
    
    function getSingValsGenericCoeffs(n, r)
        return f["$n/$r/singValsGenericCoeffs"]
    end;
    
    function getRankGenericCoeffs(n, r)
        return f["$n/$r/rankGenericCoeffs"]
    end;
    
    return getPossR, getNumVars, getNumEqs, getSingVals, getRank, getSingValsGenericCoeffs, getRankGenericCoeffs
end;