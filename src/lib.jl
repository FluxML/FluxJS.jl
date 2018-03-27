# Library of mathematical functions we consider primitive.
# TODO: store Julia functions and types, to avoid deeplearn.js-specific functions

# matmul

matVecMul(args...) = *(args...)

shape(::typeof(matVecMul), args...) = shape(*, args...)

@primitive Trace x::AbstractMatrix * y::AbstractMatrix =
  StagedArray(*, x, y)

jscall(::typeof(*), a, b) = jscall(:(math.matMul), a, b)

@primitive Trace x::AbstractMatrix * y::AbstractVector =
  StagedArray(matVecMul, x, y)

jscall(::typeof(matVecMul), a, b) = jscall(:(math.matrixTimesVector), a, b)

# cat

concat1D(a, b) = vcat(a, b)

@primitive Trace vcat(a::AbstractVector, b::AbstractVector) =
  StagedArray(concat1D, a, b)

@primitive Trace vcat(a::AbstractMatrix, b::AbstractMatrix) =
  error("concat2D not implemented")

jscall(::typeof(concat1D), a, b) = jscall(:(math.concat1D), a, b)

# softmax

@primitive Trace softmax(x::AbstractVecOrMat) =
  StagedArray(softmax, x)

jscall(::typeof(softmax), x) = jscall(:(math.softmax), x)

# broadcasted ops

bcastable(+, *, /, ^, tanh, σ, relu, leakyrelu, abs, exp, log)

jscall(::typeof(broadcast), ::typeof(+), a, b) = jscall(:(math.add), a, b)
jscall(::typeof(broadcast), ::typeof(*), a, b) = jscall(:(math.mul), a, b)
jscall(::typeof(broadcast), ::typeof(/), a, b) = jscall(:(math.div), a, b)
jscall(::typeof(broadcast), ::typeof(^), a, b) = jscall(:(math.pow), a, b)
jscall(::typeof(broadcast), ::typeof(tanh), x) = jscall(:(math.tanh), x)
jscall(::typeof(broadcast), ::typeof(σ), x) = jscall(:(math.sigmoid), x)
jscall(::typeof(broadcast), ::typeof(relu), x) = jscall(:(math.relu), x)
jscall(::typeof(broadcast), ::typeof(leakyrelu), x) = jscall(:(math.relu), x, 0.01)
jscall(::typeof(broadcast), ::typeof(leakyrelu), x, a) = jscall(:(math.relu), x, a)
jscall(::typeof(broadcast), ::typeof(abs), x) = jscall(:(math.abs), x)
jscall(::typeof(broadcast), ::typeof(exp), x) = jscall(:(math.exp), x)
jscall(::typeof(broadcast), ::typeof(log), x) = jscall(:(math.log), x)
