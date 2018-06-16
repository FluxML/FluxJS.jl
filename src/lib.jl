using NNlib: cdims, padtuple, pdims, conv

# struct Shape{T,N}
#   dims::NTuple{N,Int}
# end
#
# VecShape{T} = Shape{T,1}
# MatShape{T} = Shape{T,2}
#
# Shape{T}(dims::Vararg{Integer,N}) where {T,N} = Shape{T,N}(dims)
# Shape{T}(dims::NTuple{N,Integer}) where {T,N} = Shape{T,N}(dims)
#
# Base.size(s::Shape) = s.dims
# Base.size(s::Shape, n) = s.dims[n]
# Base.ndims(s::Shape{T,N}) where {T,N} = N
# Base.length(s::Shape) = prod(s.dims)
# Base.eltype(s::Shape{T}) where T = T
#
# Base.sizeof(s::Shape{T}) where T = sizeof(T)*prod(size(s))
# NNlib.padtuple(x::FluxJS.Shape, p) = padtuple(size(x), p)

# function Base.show(io::IO, s::Shape{T}) where T
#   print(io, "Shape{$T}(")
#   join(io, s.dims, ", ")
#   print(io, ")")
# end

# Library of mathematical functions we consider primitive.
# TODO: store Julia functions and types, to avoid deeplearn.js-specific functions

# matmul

# shape(::typeof(*), A::MatShape{T}, B::VecShape{T}) where T =
  # Shape{T}(size(A,1))

# shape(::typeof(*), A::MatShape{T}, B::MatShape{T}) where T =
  # Shape{T}(size(A,1),size(B,2))

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

# shape(::typeof(softmax), x) = shape(x)

# conv2d

@primitive Trace function (c::Conv)(x)
  out = conv(val(x), c.weight, stride = c.stride, pad = c.pad)
  pad = 0
  !all(x-> x == c.pad[1], c.pad)?
    throw(error("Assymetric padding is unsupported by deeplearn-js")):
    pad = c.pad[1]
  y = StagedArray(conv2d, x, c.weight, padtuple(x,c.stride), pad, v=out)
  σ, b = c.σ, reshape(c.bias, map(_->1, c.stride)..., :, 1)
  overdub(Trace(), (x, σ, b) -> (σ).(x .+ b), y, σ, b)
end

jscall(::typeof(conv2d), x...) = jscall(:(tf.conv2d), x...)

# shape(::typeof(conv2d), x::Shape{T}, weight, stride, pad) where T =
#   Shape{T}(cdims(size(x), size(weight), padtuple(x, pad), stride))

# maxpool

@primitive Trace function maxpool(x::AbstractArray, k; pad = map(_->0,k), stride = k)
  out = maxpool(val(x), k, pad=pad, stride=stride)

  !all(x-> x == pad[1], pad)?
    throw(error("Assymetric padding is unsupported by deeplearn-js")):
    pad = pad[1]

  StagedArray(maxpool, x, k, pad, stride, v=out)
end

jscall(::typeof(maxpool), x, k, pad, stride) = jscall(:(math.maxPool), x, k, stride, pad)

# shape(::typeof(maxpool), x::Shape{T}, k, pad, stride) where T = Shape{T}(pdims(size(x), k, padtuple(x, pad), stride))

# broadcasted ops

# shape(::typeof(broadcast), f, xs...) =
#   Shape{eltype(xs[1])}(Base.Broadcast.broadcast_shape(size.(xs)...)...)

bcastable(+, *, tanh, relu, σ)

jscall(::typeof(broadcast), ::typeof(+), a, b) = jscall(:(math.add), a, b)
jscall(::typeof(broadcast), ::typeof(σ), x) = jscall(:(math.sigmoid), x)
jscall(::typeof(broadcast), ::typeof(tanh), x) = jscall(:(math.tanh), x)
jscall(::typeof(broadcast), ::typeof(relu), x) = jscall(:(math.relu), x)

# shape(::typeof(reshape), x::Shape{T}, i...) where T =
#   Shape{T}(Base._reshape_uncolon(x, i))
#
# shape(x) = x
# shape(x::Shape) = x
# shape(x::Tuple) = shape.(x)
# shape(x::AbstractArray) = Shape{eltype(x)}(size(x)...)
# shape(x::TrackedArray) = shape(x.data)

# reshape

@primitive Trace function Base.reshape(parent::AbstractArray, dims...)
  dims = any(x -> val(x) isa Colon, dims) ?
  map((x, y) -> val(x) isa Colon? y : x
    , dims, Base._reshape_uncolon(val(parent), val.(dims))) : dims
  StagedArray(reshape, parent, dims...)
end

@primitive t::Trace function Base.reshape(parent, dims::Tuple{Vararg{Union{Int,Colon}}})
  dims = Base._reshape_uncolon(val(parent), val.(dims))
  overdub(t, (x...)-> reshape(x...), parent, dims...)
end

jscall(::typeof(reshape), p, dims...) =
  jscall(:(math.reshape), p, jscall(:(Array().constructor), dims...))

# size
@primitive Trace Base.size(x...) =
  StagedArray(size, x...)

jscall(::typeof(size), x) = jscall(:(flux.getprop), x, "shape")
jscall(s::typeof(size), x, i) = jscall(:(flux.getprop), jscall(s, x), i)
