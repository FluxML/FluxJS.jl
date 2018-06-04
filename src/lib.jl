using NNlib: cdims, padtuple

struct Shape{T,N}
  dims::NTuple{N,Int}
end

VecShape{T} = Shape{T,1}
MatShape{T} = Shape{T,2}

Shape{T}(dims::Vararg{Integer,N}) where {T,N} = Shape{T,N}(dims)
Shape{T}(dims::NTuple{N,Integer}) where {T,N} = Shape{T,N}(dims)

Base.size(s::Shape) = s.dims
Base.size(s::Shape, n) = s.dims[n]
Base.ndims(s::Shape{T,N}) where {T,N} = N
Base.length(s::Shape) = prod(s.dims)
Base.eltype(s::Shape{T}) where T = T

Base.sizeof(s::Shape{T}) where T = sizeof(T)*prod(size(s))
NNlib.padtuple(x::FluxJS.Shape, p) = padtuple(size(x), p)

function Base.show(io::IO, s::Shape{T}) where T
  print(io, "Shape{$T}(")
  join(io, s.dims, ", ")
  print(io, ")")
end

# Library of mathematical functions we consider primitive.
# TODO: store Julia functions and types, to avoid deeplearn.js-specific functions

# matmul

shape(::typeof(*), A::MatShape{T}, B::VecShape{T}) where T =
  Shape{T}(size(A,1))

shape(::typeof(*), A::MatShape{T}, B::MatShape{T}) where T =
  Shape{T}(size(A,1),size(B,2))

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

shape(::typeof(softmax), x) = shape(x)

# conv2d

@primitive Trace function (c::Conv)(x)
  pad = 0
  !all(x-> x == c.pad[1], c.pad)?
    throw(error("Assymetric padding is unsupported by deeplearn-js")):
    pad = c.pad[1]

  conv = StagedArray(conv2d, x, c.weight, padtuple(x,c.stride), pad)

  σ, b = c.σ, reshape(c.bias, 1, map(_->1, c.stride)..., :)
  overdub(Trace(), (x, σ, b) -> (σ).(x .+ b), conv, σ, b)
end

jscall(::typeof(conv2d), x...) = jscall(:(math.conv2d), x...)

shape(::typeof(conv2d), x::Shape{T,N}, weight, stride, pad) where {T,N} =
  Shape{T}(cdims(size(x), size(weight), padtuple(x, pad), stride))

# broadcasted ops

shape(::typeof(broadcast), f, xs...) =
  Shape{eltype(xs[1])}(Base.Broadcast.broadcast_shape(size.(xs)...)...)

bcastable(+, *, tanh, relu, σ)

jscall(::typeof(broadcast), ::typeof(+), a, b) = jscall(:(math.add), a, b)
jscall(::typeof(broadcast), ::typeof(σ), x) = jscall(:(math.sigmoid), x)
jscall(::typeof(broadcast), ::typeof(tanh), x) = jscall(:(math.tanh), x)
jscall(::typeof(broadcast), ::typeof(relu), x) = jscall(:(math.relu), x)

shape(::typeof(reshape), x::Shape{T}, i...) where T =
  Shape{T}(Base._reshape_uncolon(x, i))

shape(x) = x
shape(x::Shape) = x
shape(x::Tuple) = shape.(x)
shape(x::AbstractArray) = Shape{eltype(x)}(size(x)...)
shape(x::TrackedArray) = shape(x.data)
