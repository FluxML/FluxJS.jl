# Library of mathematical functions we consider primitive.

@primitive Trace (::typeof(*))(x::AbstractMatrix, y::AbstractVecOrMat) =
  StagedArray{Real,ndims(y)}(*, x, y)

bcastable(+, *)

# f(a, b) = a*b .+ b
# trace(f, Array{Real,2}, Array{Real,1})
