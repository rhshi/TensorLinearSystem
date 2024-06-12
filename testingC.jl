include("testing2.jl");

function tupLT(a, b)
    if a[1][1] isa Number && !(b[1][1] isa Number)
        return false
    elseif !(a[1][1] isa Number) && b[1][1] isa Number
        return true
    else
        return a < b
    end
end;

function tupLT2(a, b)
    if a[1][1] isa Number && !(b[1][1] isa Number)
        if a == b[1]
            return false
        else
            return a < b[1]
        end
    elseif !(a[1][1] isa Number) && b[1][1] isa Number
        if a[1] == b
            return true
        else
            return a[1] < b
        end
    elseif !(a[1][1] isa Number) && !(b[1][1] isa Number)
        if (a[1][1] == b[1][1])
            if a[2][1] == b[2][1]
                return a < b
            else
                return a[2][1] < b[2][1]
            end
        else
            return a[1][1] < b[1][1]
        end
    else
        return a < b
    end
end;

function makeVarLT(c)
    function varLT(a, b)
        ac = sum([x for x in a][1:c])
        bc = sum([x for x in b][1:c])

        if ac == bc
            return a < b
        else
            return ac < bc

        end
    end
    return varLT
end;

function makeSum(c)
    function getl(k)
        out = 0
        for j = k:c
            out += n - j + 1
        end
        return out
    end


    function getL(k)
        out = 0
        for j = k:c
            out += getl(j)
        end
        return out
    end

    function sumL(k)
        out = 0
        for j = 1:k
            out += getL(j)
        end
        return out
    end

    return getl, getL, sumL
end;

function getNonDuplicates(eqTups)
    nonDuplicateTups = []
    nonDuplicateInds = []
    for (i, eqTup) in enumerate(eqTups)
        if !(eqTup in nonDuplicateTups)
            push!(nonDuplicateInds, i)
            push!(nonDuplicateTups, eqTup)
        end
    end
    return nonDuplicateTups, nonDuplicateInds
end;

function getNonActive(eqTups, numActiveEqs)
    nonActiveInds = []
    for (i, eqTup) in enumerate(eqTups[numActiveEqs+1:end])
        alpha_i = eqTup[1][1]
        if sum([x for x in alpha_i][1:c]) == 1
            push!(nonActiveInds, numActiveEqs + i)
        end
    end
    return nonActiveInds
end;

function linearSystem(A)
    n_, r = size(A)
    n = n_-1
    A2_ = khatri_rao(A, 2)
    T = reshape(A2_ * permutedims(A2_), (n+1, n+1, n+1, n+1))

    X, Y, varTups, eqTups, basis, abij, h = linearSystemLarge(T, r)

    nonDuplicateTups, nonDuplicateInds = getNonDuplicates(eqTups)
    perm = sortperm(nonDuplicateTups, lt=tupLT, rev=true)
    X = X[nonDuplicateInds[perm], :]
    Y = Y[nonDuplicateInds[perm], :]
    eqTups = eqTups[nonDuplicateInds[perm]]

    return X, Y, varTups, eqTups, h
end;

function linearSystemC(A, c)
    X, Y, varTups, eqTups, h = linearSystem(A);

    n_, r = size(A)
    n = n_-1
    
    numActiveEqs = binomial(n-c+1, 2) * binomial(c+2, 3) + (n-c) * binomial(c+1, 2) * binomial(n-c+1, 2)

    nonActiveInds = getNonActive(eqTups, numActiveEqs)
    X = X[vcat(collect(1:numActiveEqs), nonActiveInds), :]
    Y = Y[vcat(collect(1:numActiveEqs), nonActiveInds), :]
    eqTups = eqTups[vcat(collect(1:numActiveEqs), nonActiveInds)]

    return X, Y, varTups, eqTups, h
end;

function findSubmatrix(c)
    k1 = Int((n - c + 1) * (n - c) / 2)
    k2 = Int((n - c + 1) * (n - c) * (n - c - 1) / 3)

    l, L, sumL = makeSum(c)

    inds = []

    l1 = 0
    l2 = 0
    inds = []
    start = 0
    totalL = 0

    for i = 1:c
        for j = i:c
            l1 += c - j + 1
            l2 += 1
            totalL += L(j)

            stop = l1 * k1 + (l2 - 1) * k2 + min(l2 * k2, totalL) - min((l2 - 1) * k2, totalL - L(j))

            append!(inds, collect(start+1:stop))
            start = l1 * k1 + l2 * k2
        end
    end

    return inds
end

function getSquareMat(A, c)
    n_, _ = size(A)
    n = n_-1

    X, Y, varTups, eqTups, h = linearSystemC(A, c);
    
    numActiveEqs = binomial(n-c+1, 2) * binomial(c+2, 3) + (n-c) * binomial(c+1, 2) * binomial(n-c + 1, 2);
    X = X[1:numActiveEqs, :]
    Y = Y[1:numActiveEqs, :]
    eqTups = eqTups[1:numActiveEqs];
 
    lastVarDict = Dict()
    for (i, eqTup) in enumerate(eqTups)
        lastVar = [x for x in eqTup[1]] + [x for x in eqTup[2]]
        if !(lastVar in keys(lastVarDict))
            lastVarDict[lastVar] = [i]
        else 
            push!(lastVarDict[lastVar], i)
        end 
    end

    Xrows = []
    Yrows = []
    eqTupsNew = []

    for lastVar in keys(lastVarDict)
        if length(lastVarDict[lastVar]) == 1
            i = lastVarDict[lastVar][1]
            if sum([x for x in eqTups[i][1]][1:c]) == 3
                push!(Xrows, X[i, :])
                push!(Yrows, Y[i, :])
                push!(eqTupsNew, eqTups[i])
            end
        elseif length(lastVarDict[lastVar]) == 2
            i = lastVarDict[lastVar][1]
            j = lastVarDict[lastVar][2]
            push!(Xrows, X[i, :]-X[j, :])
            push!(Yrows, Symbolics.expand.(Y[i, :]-Y[j, :]))
            push!(eqTupsNew, (eqTups[i], eqTups[j]))
        elseif length(lastVarDict[lastVar]) == 3
            i = lastVarDict[lastVar][1]
            j = lastVarDict[lastVar][2]
            k = lastVarDict[lastVar][3]
            push!(Xrows, X[i, :]-X[j, :])
            push!(Yrows, Symbolics.expand.(Y[i, :]-Y[j, :]))
            push!(eqTupsNew, (eqTups[i], eqTups[j]))
            push!(Xrows, X[i, :]-X[k, :])
            push!(Yrows, Symbolics.expand.(Y[i, :]-Y[k, :]))
            push!(eqTupsNew, (eqTups[i], eqTups[k]))
        end 
    end 

    perm = sortperm(eqTupsNew, lt=tupLT2, rev=true)
    eqTups = eqTupsNew[perm];
    X = permutedims(reduce(hcat, Xrows))[perm, :]
    Y = permutedims(reduce(hcat, Yrows))[perm, :];

    X_ = X[:, findall(!iszero, eachcol(X))]
    Y_ = Y[:, findall(!iszero, eachcol(X))];

    inds = findSubmatrix(c);

    return X_[inds, :], Y_[inds, :], varTups[findall(!iszero, eachcol(X))], eqTups[inds], h
 
end;
