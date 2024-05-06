include("util.jl")

function makeRankedTensor(L::Vector, A::Array, d::Int)
    n, _ = size(A)
    A_ = khatri_rao(A, d)
    return reshape(sum(transpose(L) .* A_, dims=2), tuple(repeat([n], d)...))
end;

function randomRankedTensor(n, d, r; real=false)
    if !real
        A = complexGaussian(n, r)
    else 
        A = randn(n, r)
    end;
    L = ones(r)
    A_ = copy(A)
    A1 = A_[1, :]
    L_ = zeros(eltype(A_), r)
    for i=1:r
        A_[:, i] ./= A1[i]
        L_[i] = A1[i]^d
    end;
    signs = rand([-1, 1], r)
    return makeRankedTensor(L .* signs, A, d), A_, L_ .* signs
end;