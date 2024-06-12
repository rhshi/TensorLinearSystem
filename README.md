# TensorLinearSystem

Dependencies:
* LinearAlgebra
* Combinatorics
* InvertedIndices
* Printf
* JLD2
* Symbolics (if wanting to analyze some of these matrices symbolically)

The main script is `testScript.jl`.  Just run `julia testScript.jl` and you'll receive three input prompts: the first one will be to enter the file name, e.g., if you type in "results" the experiment will be saved to a file called `results/results.jld2`.  The next prompt will be to enter the $n$ we want to start at and the last will be to enter the n we want to end at, i.e., if you type in "2" and then "10" the experiment will run for $n=2,...,10$.  We test all $r$ from $n+2$ to $\binom{n+2}{2}$, but as soon as the number of equations becomes less than the number of variables we break the loop and increase n.  This is just for performance reasons.

In the saved data there is one field specific to $n$:
1) "n/numR": the ranks that are active for $n$, in the sense that for these $r$ the number of equations is at least the number of variables

And six fields specific to $n$ and $r$, for each $r$ in the above range:
1) "n/r/numVars": number of variables
2) "n/r/numEqs": number of equations
3) "n/r/singVals": singular values of the coefficient matrix
4) "n/r/rank": rank obtained by a thresholding of the above singular values
5) "n/r/singValsGenericCoeffs": singular values of the coefficient matrix after replacing the entries with generic values, one for each minor
6) "n/r/rankGenericCoeffs": rank given by a thresholding of the previous generic coefficient matrix

To access the above data I recommend opening a Jupyter notebook.  Then run
```julia
include("testing.jl")
f = jldopen($filename$, "r") # $filename$ is the file name, e.g., "results/results.jld2"
getPossR, getNumVars, getNumEqs, getSingVals, getRank, getSingValsGenericCoeffs, getRankGenericCoeffs = makeStats(f)
```
These are functions that take in $n$ (and possibly $r$) and return the data above.  `getPossR` is just a function of $n$, and the rest are functions of $n$ and $r$.
