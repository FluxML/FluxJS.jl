# Library of mathematical functions we consider primitive.

matVecMul(args...) = *(args...)

@primitive Trace (::typeof(*))(x::AbstractMatrix, y::AbstractMatrix) =
  StagedArray{Real,2}(*, x, y)

@primitive Trace (::typeof(*))(x::AbstractMatrix, y::AbstractVector) =
  StagedArray{Real,1}(matVecMul, x, y)

bcastable(+, *)

# f(a, b) = a*b .+ b
# trace(f, Array{Real,2}, Array{Real,1})
