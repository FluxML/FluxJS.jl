using NNlib: cdims, padtuple, pdims, conv
using Vinyl: primitive

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
jscall(::typeof(broadcast), ::typeof(*), x, y) = jscall(:(math.matMul), x, y)

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

jscall(::typeof(reshape), p, dims...) =
  all(x -> x isa Integer, dims) ? (jscall(:(math.reshape), p, dims)) :
  jscall(:(math.reshape), p, jscall(:(Array().constructor), dims...))

# size
@primitive Trace Base.size(x) =
  StagedArray(getindex, x, "shape", v=size(val(x)))

@primitive Trace function Base.size(x, i)
  _size = overdub(Trace(), (x) -> size(x), x);
  index = overdub(Trace(), (i)-> (i - 1), i)
  StagedArray(getindex, _size, index, v=size(val(x))[val(i)])
end

@primitive ctx::Trace function Flux.gate(x::AbstractArray, h, n)
  out = Flux.gate(val(x), val(h), val(n))
  _start =  overdub(Trace(), (h, n) -> h * (n-1), h, n)
  StagedArray(view, x, _start, h, v=out)
end

jscall(::typeof(view), x, start, length) =
  jscall(:(math.slice), x, start, length)

@primitive Trace start(x::StagedArray) = StagedArray(start, x)

@primitive Trace function Base.getindex(t::StagedArray, i::Union{Int,StagedArray{Int}})
  index =  primitive(Trace(), (-), i, 1)
  StagedArray(getindex, t, index, v = val(t)[val(i)])
end

# @primitive Trace function Base.getindex(t::StagedArray, i...)
#   _begin = []
#   _size = []
#   for j=1:length(i)
#     b, s = split(t, i[j], j)
#     push!(_begin, b)
#     push!(_size, s)
#   end
#   StagedArray(view, t, _begin, _size, v=getindex(val(t), val.(i)...))
# end
#
# Base.split(t, i::Union{Int,StagedArray{Int}}, j) = (val(i) - 1, 1)
# function Base.split(t, i::StagedArray{UnitRange{Int}}, j)
#   start = StagedArray{Int,0}(graph(i).inputs[1], val(i).start)
#   stop = StagedArray{Int,0}(graph(i).inputs[2], val(i).stop)
#   @show graph(start)
#   @show graph(stop)
#   c = primitive(Trace(), (-), start, 1)
#   (c , primitive(Trace(),(-), stop, c)) # (start - 1, stop - start + 1)
# end
# Base.split(t, ::Colon, j) = (0, primitive(Trace(), size, t, j))
#
# function jscall(::typeof(view), t, _begin, _size)
#   println("view")
#   @show t, _begin, _size
#   jscall(:(math.slice), t, jscall(:(Array().constructor), _begin...), jscall(:(Array().constructor), _size...))
# end
#
# @primitive Trace (f::Any)(t::StagedArray{UnitRange{Int}}) = StagedArray(f, t)
#
# Base.start(::StagedArray{T,N}) where {T<:Union{AbstractArray,Tuple},N} = 1
#
# Base.convert(::Type{StagedArray{T,N}}, x::StagedArray{T,N}) where {T,N} = x
# Base.convert(::Type{StagedArray{T,N}}, x::StagedArray{S,N}) where {T,S,N} = StagedArray{T,N}(graph(x),convert(T,val(x)))
#
# # for StagedArray{Tuple}
#
# @primitive Trace Base.start(t::StagedArray{Tuple{T},N}) where T where N =
#   stage(1, DataFlow.constant(1))
#
# @primitive Trace Base.length(t::StagedArray{Tuple{T},N}) where T where N =
#   StagedArray(length, t)

add(x, y) = x + y
sub(x, y) = x - y
mul(x, y) = x * y

# for StagedArray{Int}
function binary_op(op, sub)
  @eval @primitive Trace ($op)(x::T, y::T) where {T<:StagedArray{Int64,0}} = StagedArray($sub, x, y)
  @eval @primitive Trace ($op)(x::T, y::S) where {T<:StagedArray{Int64,0}} where {S<:Int64} = StagedArray($sub, x, y)
  @eval @primitive Trace ($op)(x::S, y::T) where {T<:StagedArray{Int64,0}} where {S<:Int64} = StagedArray($sub, x, y)
end

binary_op(+, add)
binary_op(*, mul)
binary_op(-, sub)

jscall(::typeof(add), x, y) = jscall(:(flux.add), x, y)
# jscall(::typeof(-), x::Union{StagedArray{Int},Int}, y::Union{StagedArray{Int},Int}) = jscall(:(flux.sub), x, y)
jscall(::typeof(mul), x, y) = jscall(:(flux.mul), x, y)
