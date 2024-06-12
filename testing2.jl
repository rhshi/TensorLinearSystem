include("util.jl") 
using Symbolics;

function linEqsTupsSmall(alphas, betas, c, n, D, basis, basis_inds)
    eqTups = []
    abij = []

    for alpha_ in alphas
        alpha = [a_ for a_ in alpha_]
        if sum(alpha[2:c+1]) == 2
            for beta_ in betas
                beta = [b_ for b_ in beta_]
                if findfirst(!iszero, beta[2:end]) > c
                    for i=1:n-1
                        ei = e(i+1, n)
                        ei[1] = -1
                        if i <= c 
                            for j=c+1:n-1
                                ej = e(j+1, n)
                                ej[1] = -1

                                a_i = Tuple((alpha+ei))
                                b_j = Tuple((beta+ej))
                                
                                push!(eqTups, ((a_i[2:end], b_j[2:end]), 1))
                                push!(abij, (alpha[2:end], beta[2:end], i, j))
                            end
                        else 
                            for j=i+1:n-1
                                ej = e(j+1, n)
                                ej[1] = -1

                                a_i = Tuple((alpha+ei))
                                a_j = Tuple((alpha+ej))
                                b_i = Tuple((beta+ei))
                                b_j = Tuple((beta+ej))
                                
                                push!(eqTups, (((a_i[2:end], b_j[2:end]), (a_j[2:end], b_i[2:end])), 2))
                                push!(abij, (alpha[2:end], beta[2:end], i, j))
                            end
                        end
                    end
                end
            end
        end
    end
    
    return eqTups, abij
end;

function linEqsTupsLarge(set1, set2, n, D, basis, basis_inds, twoSide=true)
    eqTups = []
    abij = []

    for alpha_ in set1
        alpha = [a_ for a_ in alpha_]
        for beta_ in set2
            beta = [b_ for b_ in beta_]
            for i=1:n-1
                ei = e(i+1, n)
                ei[1] = -1
                for j=i+1:n-1
                    ej = e(j+1, n)
                    ej[1] = -1


                    a_i = Tuple((alpha+ei))
                    a_j = Tuple((alpha+ej))
                    b_i = Tuple((beta+ei))
                    b_j = Tuple((beta+ej))
                    
                    if (b_i in basis) && !(b_j in basis) 
                        push!(eqTups, ((a_i[2:end], b_j[2:end]), 1))
                        push!(abij, (alpha[2:end], beta[2:end], i, j))
                    elseif (b_j in basis) && !(b_i in basis)
                        push!(eqTups, ((a_j[2:end], b_i[2:end]), 1))
                        push!(abij, (alpha[2:end], beta[2:end], i, j))
                    elseif !(b_i in basis) && !(b_j in basis) 
                        if twoSide
                            push!(eqTups, (((a_i[2:end], b_j[2:end]), (a_j[2:end], b_i[2:end])), 2))
                            push!(abij, (alpha[2:end], beta[2:end], i, j))
                        end
                    end

                    

                end
            end
        end
    end
    
    return eqTups, abij
end;


function getVarEqTupsSmall(alphas, betas, c, n, D, basis, basis_inds)
    varTups = varsInds(alphas, D);
    varTups = sort([var for var in varTups], rev=true)
    
    eqTups, abij = linEqsTupsSmall(alphas, betas, c, n+1, D, basis, basis_inds);

    return varTups, eqTups, abij
end

function getVarEqTupsLarge(alphas, betas, n, D, basis, basis_inds)
    varTups = varsInds(alphas, D);
    varTups = sort([var for var in varTups], rev=true)
    
    eqTups, abij = linEqsTupsLarge(alphas, betas, n+1, D, basis, basis_inds);

    return varTups, eqTups, abij
end




function linearSystemSmall(T, c)
    n = size(T)[1]-1
    d = length(size(T))

    r = Int(1+(c+1)*n+c*(1-c)/2)
    basis_inds = collect(1:r);
    
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

    varTups, eqTups, abij = getVarEqTupsSmall(alphas, betas, c, n, D, basis, basis_inds)
    
    coeffDict = Dict()
    for (i, varTup) in enumerate(varTups)
        coeffDict[varTup] = i
    end

    b_js = Set()
    for eqTupPair in eqTups
        eqTup = eqTupPair[1]
        flag = eqTupPair[2]

        if flag == 1
            b_j = eqTup[2]
            push!(b_js, b_j)
        elseif flag == 2
            b_j1 = eqTup[1][2]
            b_j2 = eqTup[2][2]
            push!(b_js, b_j1)
            push!(b_js, b_j2)
        end 
    end
    b_js = sort([b_j for b_j in b_js], rev=true)

    @variables h[0:length(alphas)*length(b_js)]
    h = Symbolics.scalarize(h)
    tempCounter = 1
    genericDict = Dict()
    for alpha in alphas
        for b_j in b_js
            genericDict[(alpha[2:end], b_j)] = h[tempCounter+1]
            tempCounter += 1
        end
    end
    detValue = h[1]

    X = Array{Complex, 2}(undef, (length(eqTups), length(varTups)))
    Y = Array{Num, 2}(undef, (length(eqTups), length(varTups)))

    for (k, eqTupPair) in enumerate(eqTups)
        eqTup = eqTupPair[1]
        flag = eqTupPair[2]

        if flag == 1
            coeffRow, _ = processLinEq(eqTup, D, coeffDict, alphas, basis_inds, H0_det, H0_adj, TcatGam, TcatDelt, d)
            X[k, :] = coeffRow

            coeffRow2 = processLinEqGeneric(eqTup, coeffDict, genericDict, alphas, d, detValue; numType=Num)
            Y[k, :] = coeffRow2
        elseif flag == 2
            eqTupL = eqTup[1]
            eqTupR = eqTup[2]

            coeffRowL, _ = processLinEq(eqTupL, D, coeffDict, alphas, basis_inds, H0_det, H0_adj, TcatGam, TcatDelt, d)
            coeffRowR, _ = processLinEq(eqTupR, D, coeffDict, alphas, basis_inds, H0_det, H0_adj, TcatGam, TcatDelt, d)

            X[k, :] = coeffRowL - coeffRowR

            coeffRowL2 = processLinEqGeneric(eqTupL, coeffDict, genericDict, alphas, d, detValue; numType=Num)
            coeffRowR2 = processLinEqGeneric(eqTupR, coeffDict, genericDict, alphas, d, detValue; numType=Num)

            Y[k, :] = Symbolics.expand.(coeffRowL2 - coeffRowR2)
        end
        
    end

    X ./= sqrt(Complex.(det(H0)))

    return X, Y, varTups, first.(eqTups), [b[2:end] for b in basis], abij, h
        
    
end;


function linearSystemLarge(T, r)
    n = size(T)[1]-1
    d = length(size(T))

    basis_inds = collect(1:r);
    
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

    varTups, eqTups, abij = getVarEqTupsLarge(alphas, betas, n, D, basis, basis_inds)
    
    coeffDict = Dict()
    for (i, varTup) in enumerate(varTups)
        coeffDict[varTup] = i
    end

    b_js = Set()
    for eqTupPair in eqTups
        eqTup = eqTupPair[1]
        flag = eqTupPair[2]

        if flag == 1
            b_j = eqTup[2]
            push!(b_js, b_j)
        elseif flag == 2
            b_j1 = eqTup[1][2]
            b_j2 = eqTup[2][2]
            push!(b_js, b_j1)
            push!(b_js, b_j2)
        end 
    end
    b_js = sort([b_j for b_j in b_js], rev=true)

    @variables h[0:length(alphas)*length(b_js)]
    h = Symbolics.scalarize(h)
    tempCounter = 1
    genericDict = Dict()
    for alpha in alphas
        for b_j in b_js
            genericDict[(alpha[2:end], b_j)] = h[tempCounter+1]
            tempCounter += 1
        end
    end
    detValue = h[1]

    X = Array{Complex, 2}(undef, (length(eqTups), length(varTups)))
    Y = Array{Num, 2}(undef, (length(eqTups), length(varTups)))


    for (k, eqTupPair) in enumerate(eqTups)
        eqTup = eqTupPair[1]
        flag = eqTupPair[2]

        if flag == 1
            coeffRow, _ = processLinEq(eqTup, D, coeffDict, alphas, basis_inds, H0_det, H0_adj, TcatGam, TcatDelt, d)
            X[k, :] = coeffRow

            coeffRow2 = processLinEqGeneric(eqTup, coeffDict, genericDict, alphas, d, detValue; numType=Num)
            Y[k, :] = coeffRow2
        elseif flag == 2
            eqTupL = eqTup[1]
            eqTupR = eqTup[2]

            coeffRowL, _ = processLinEq(eqTupL, D, coeffDict, alphas, basis_inds, H0_det, H0_adj, TcatGam, TcatDelt, d)
            coeffRowR, _ = processLinEq(eqTupR, D, coeffDict, alphas, basis_inds, H0_det, H0_adj, TcatGam, TcatDelt, d)

            X[k, :] = coeffRowL - coeffRowR

            coeffRowL2 = processLinEqGeneric(eqTupL, coeffDict, genericDict, alphas, d, detValue; numType=Num)
            coeffRowR2 = processLinEqGeneric(eqTupR, coeffDict, genericDict, alphas, d, detValue; numType=Num)

            Y[k, :] = Symbolics.expand.(coeffRowL2 - coeffRowR2)
        end
        
    end

    X ./= sqrt(Complex.(det(H0)))

    return X, Y, varTups, first.(eqTups), [b[2:end] for b in basis], abij, h

end