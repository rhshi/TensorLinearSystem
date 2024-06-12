using LinearAlgebra, Combinatorics, InvertedIndices


function khatri_rao(A, d)
    if d == 1
        B = A
    else
        n, r = size(A)
        B = zeros(eltype(A), (n^d, r))
        for i=1:r 
            B[:, i] = kron(ntuple(x->A[:, i], d)...)
        end;
    end;
    return B
end

function krDrop(A, d)
    n = size(A)[1]-1
    inds = collect(with_replacement_combinations(1:n+1, d))
    inds = map(x->from_multiindex(x, n+1), inds)
    return khatri_rao(A, d)[inds, :]
end;

function makeDicts(n, d)
    monomials = reverse(sort(collect(alpha_iterator(Val(n), Int(d)))));
    D = Dict()
    Drev = Dict()
    for (i, mon) in enumerate(monomials)
        D[mon] = i 
        Drev[i] = mon
    end 
    return D, Drev
end

function alpha_iterator(::Val{N}, s, t=()) where {N}
    N <= 1 && return ((s, t...),) # Iterator over a single Tuple
    Iterators.flatten(alpha_iterator(Val(N-1), s-i, (i, t...)) for i in 0:s)
end

function basisFn(basis_inds, Drev, d)
    basis = [Drev[ind] for ind in basis_inds];
    
    basisD = Dict()
    for (i, b) in enumerate(basis)
        if d-b[1] in keys(basisD)
            push!(basisD[d-b[1]], i)
        else
            basisD[d-b[1]] = [i]
        end
    end
    return basis, basisD
end;

function from_multiindex(x, n)
    d = length(x)
    c = 0
    for i=1:d-1
        c += (x[i]-1)*n^(d-i)
    end
    return c + x[d]
end;

function catMat(T, k)
    Tsize = size(T)
    n = Tsize[1]
    d = length(Tsize)

    low = k 
    high = d-k 

    row_inds = collect(with_replacement_combinations(1:n, low))
    row_inds = map(x->from_multiindex(x, n), row_inds)

    col_inds = collect(with_replacement_combinations(1:n, high))
    col_inds = map(x->from_multiindex(x, n), col_inds)

    return (reshape(T, (n^low, n^high)))[row_inds, col_inds]
end;

function cofactor(A::AbstractMatrix)
    ax = axes(A)
    out = similar(A, eltype(A), ax)
    for col in ax[1]
        for row in ax[2]
            out[col, row] = (-1)^(col + row) * det(A[Not(col), Not(row)])
        end
    end
    return out
end

function varsInds(alphas, D)
    n = length(alphas[1])
    vars = Set()
    Id = hcat(-ones(Int64, n-1), diagm(ones(Int64, n-1)))
    for alpha in alphas
        alpha = [x_ for x_ in alpha]
        for alpha_ in alphas
            for j=1:n-1
                ej = Id[j, :]
                alpha_j = [x_ for x_ in alpha_]+ej

                push!(vars, Tuple((alpha+alpha_j)[2:end]))
            end
        end
    end
    return vars
    
end

function e(j, n)
    ej = zeros(Int64, n)
    ej[j] = 1
    return ej
end

function makeEqsTups(set1, set2, n, D, basis, basis_inds; linear=true)
    eqTups1 = Set()
    eqTups2 = Set()

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
                    
                    if linear
                        if (b_i in basis) && !(b_j in basis) 
                            push!(eqTups1, (a_i[2:end], b_j[2:end]))
                        elseif (b_j in basis) && !(b_i in basis)
                            push!(eqTups1, (a_j[2:end], b_i[2:end]))
                        elseif !(b_i in basis) && !(b_j in basis) 
                            push!(eqTups2, ((a_i[2:end], b_j[2:end]), (a_j[2:end], b_i[2:end])))
                        end
                    else
                        push!(eqTups1, ((a_i[2:end], b_j[2:end]), (a_j[2:end], b_i[2:end])))
                    end

                end
            end
        end
    end
    
    return eqTups1, eqTups2
end;

function processLinEq(eqTup, D, coeffDict, alphas, basis_inds, H0_det, H0_adj, Tcat, Tcat1, d)
    coeffRow = zeros(Complex, length(coeffDict))
    
    a_i = [x for x in eqTup[1]]
    b_j = [x for x in eqTup[2]]
    
    a_i_ = Tuple(vcat(d-sum(a_i), a_i))
    b_j_ = Tuple(vcat(d-sum(b_j), b_j))
    
    coeffs = H0_adj*Tcat[basis_inds, D[b_j_]] 
    active_coeffs = coeffs[end-length(alphas)+1:end]
    nonactive_coeffs = coeffs[1:end-length(alphas)]
    
    if Tuple(a_i+b_j) in keys(coeffDict)
        coeffRow[coeffDict[Tuple(a_i+b_j)]] = H0_det
    end
    
    for (i, alpha_prime_) in enumerate(alphas)
        alpha_prime = [x for x in alpha_prime_[2:end]]
        coeffRow[coeffDict[Tuple(a_i+alpha_prime)]] = -active_coeffs[i]
    end
    
    constant = dot(Tcat1[:, D[a_i_]], nonactive_coeffs)
    
    return coeffRow, constant
    
end

function processLinEqGeneric(eqTup, coeffDict, genericDict, alphas, d, detValue; numType=Float64)
    coeffRow = zeros(numType, length(coeffDict))
    
    a_i = [x for x in eqTup[1]]
    b_j = [x for x in eqTup[2]]
    
    if Tuple(a_i+b_j) in keys(coeffDict)
        coeffRow[coeffDict[Tuple(a_i+b_j)]] = detValue
    end
    
    for (i, alpha_prime_) in enumerate(alphas)
        alpha_prime = [x for x in alpha_prime_[2:end]]
        coeffRow[coeffDict[Tuple(a_i+alpha_prime)]] = (-1)^i * genericDict[(Tuple(alpha_prime), Tuple(b_j))]
    end
    
    return coeffRow
    
end

function linEqsTups(set1, set2, n, D, basis, basis_inds, twoSide=true)
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