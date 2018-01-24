# Library of mathematical functions we consider primitive.
# TODO: store Julia functions and types, to avoid deeplearn.js-specific functions

# matmul

matVecMul(args...) = *(args...)

@primitive Trace (::typeof(*))(x::AbstractMatrix, y::AbstractMatrix) =
  StagedArray{Real,2}(*, x, y)

jscall(::typeof(*), a, b) = jscall(:(math.matMul), a, b)

@primitive Trace (::typeof(*))(x::AbstractMatrix, y::AbstractVector) =
  StagedArray{Real,1}(matVecMul, x, y)

jscall(::typeof(matVecMul), a, b) = jscall(:(math.matrixTimesVector), a, b)

# cat

concat1D(a, b) = vcat(a, b)

@primitive Trace (::typeof(vcat))(a::AbstractVector, b::AbstractVector) =
  StagedArray{Real,1}(concat1D, a, b)

@primitive Trace (::typeof(vcat))(a::AbstractMatrix, b::AbstractMatrix) =
  error("concat2D not implemented")

jscall(::typeof(concat1D), a, b) = jscall(:(math.concat1D), a, b)

# softmax

@primitive Trace (::typeof(softmax))(x::AbstractVecOrMat) =
  StagedArray{eltype(x),ndims(x)}(softmax, x)

jscall(::typeof(softmax), x) = jscall(:(math.softmax), x)

# broadcasted ops

bcastable(+, *, tanh)

jscall(::typeof(broadcast), ::typeof(+), a, b) = jscall(:(math.add), a, b)
jscall(::typeof(broadcast), ::typeof(σ), x) = jscall(:(math.sigmoid), x)
jscall(::typeof(broadcast), ::typeof(tanh), x) = jscall(:(math.tanh), x)
