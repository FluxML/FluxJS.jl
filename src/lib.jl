using NNlib: cdims, padtuple, pdims

const to_NCHW = :([0, 3, 1, 2])
const to_NHWC = :([0, 2, 3, 1])

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
# TODO: store Julia functions and types, to avoid tensorflow.js-specific functions

# matmul

shape(::typeof(*), A::MatShape{T}, B::VecShape{T}) where T =
  Shape{T}(size(A,1))

shape(::typeof(*), A::MatShape{T}, B::MatShape{T}) where T =
  Shape{T}(size(A,1),size(B,2))

matVecMul(args...) = *(args...)

shape(::typeof(matVecMul), args...) = shape(*, args...)

@primitive Trace x::AbstractMatrix * y::AbstractMatrix =
  StagedArray(*, x, y)

jscall(::typeof(*), a, b) = jscall(:(math.matMul), b, a)

@primitive Trace x::AbstractMatrix * y::AbstractVector =
  StagedArray(matVecMul, x, y)

jscall(::typeof(matVecMul), a, b) = jscall(:(math.vectorTimesMatrix), b, a)

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

  y = StagedArray(conv2d, stagedinputs(x)..., c.weight, padtuple(x,c.stride), pad)
  σ, b = c.σ, reshape(c.bias, map(_->1, c.stride)..., :, 1)
  out = overdub(Trace(), (x) -> (σ).(x .+ b), y)
  wrap(out, vcall(vertex(DataFlow.Lambda(1, unwrap(out))), x))
end

Base.permutedims(x::Union{StagedArray,IVertex}, p) = jscall(:(math.transpose), x, p)
Base.reverse(x::StagedArray) = jscall(:(math.transpose), x)

# tf-js uses NHWC while js default is NCHW
function jscall(::typeof(conv2d), x, w, s, p)
  _x = permutedims(x, to_NHWC)
  _w = jscall(:(math.reverse), jscall(:(math.transpose), w, :([2, 3, 1,0]), x), :([0,1]))
  _s = reverse(s)
  _out = jscall(:(math.conv2d), _x, _w, _s, p)
  permutedims(_out, to_NCHW)
end

shape(::typeof(conv2d), x::Shape{T}, weight, stride, pad) where T =
  Shape{T}(cdims(size(x), size(weight), padtuple(x, pad), stride))

# maxpool

@primitive Trace function maxpool(x::AbstractArray, k; pad = map(_->0,k), stride = k)
  !all(x-> x == pad[1], pad)?
    throw(error("Assymetric padding is unsupported by deeplearn-js")):
    pad = pad[1]

  StagedArray(maxpool, x, k, pad, stride)
end

function jscall(::typeof(maxpool), x, k, pad, stride)
  _x = permutedims(x, to_NHWC)
  _k = reverse(k)
  _s = reverse(stride)
  _out = jscall(:(math.maxPool), _x, _k, _s, pad)
  permutedims(_out, to_NCHW)
end

shape(::typeof(maxpool), x::Shape{T}, k, pad, stride) where T =
 Shape{T}(pdims(size(x), k, padtuple(x, pad), stride))

# broadcasted ops

shape(::typeof(broadcast), f, xs...) =
  Shape{eltype(xs[1])}(Base.Broadcast.broadcast_shape(size.(xs)...)...)

bcastable(+, *, /, ^, tanh, σ, relu, leakyrelu, abs, exp, log)

jscall(::typeof(broadcast), ::typeof(+), a, b) = jscall(:(math.add), a, b)
jscall(::typeof(broadcast), ::typeof(*), a, b) = jscall(:(math.mul), a, b)
jscall(::typeof(broadcast), ::typeof(/), a, b) = jscall(:(math.div), a, b)
jscall(::typeof(broadcast), ::typeof(^), a, b) = jscall(:(math.pow), a, b)
jscall(::typeof(broadcast), ::typeof(tanh), x) = jscall(:(math.tanh), x)
jscall(::typeof(broadcast), ::typeof(σ), x) = jscall(:(math.sigmoid), x)
jscall(::typeof(broadcast), ::typeof(relu), x) = jscall(:(math.relu), x)
jscall(::typeof(broadcast), ::typeof(leakyrelu), x) = jscall(:(math.leakyRelu), x, 0.01)
jscall(::typeof(broadcast), ::typeof(leakyrelu), x, a) = jscall(:(math.leakyRelu), x, a)
jscall(::typeof(broadcast), ::typeof(abs), x) = jscall(:(math.abs), x)
jscall(::typeof(broadcast), ::typeof(exp), x) = jscall(:(math.exp), x)
jscall(::typeof(broadcast), ::typeof(log), x) = jscall(:(math.log), x)

shape(::typeof(reshape), x::Shape{T}, i...) where T =
  Shape{T}(Base._reshape_uncolon(x, i))

shape(x) = x
shape(x::Shape) = x
shape(x::Tuple) = shape.(x)
shape(x::AbstractArray) = Shape{eltype(x)}(size(x)...)
shape(x::TrackedArray) = shape(x.data)
