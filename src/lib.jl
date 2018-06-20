using NNlib: cdims, padtuple, pdims, conv
using Vinyl: primitive

const to_NCHW = :([0, 3, 1, 2])
const to_NHWC = :([0, 2, 3, 1])

# Library of mathematical functions we consider primitive.
# TODO: store Julia functions and types, to avoid deeplearn.js-specific functions

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

Base.permutedims(x::Union{StagedArray,IVertex}, p) = jscall(:(math.transpose), x, p)
Base.reverse(x::StagedArray) = jscall(:(math.transpose), x)

# tf-js uses NHWC while js default is NCHW
function jscall(::typeof(conv2d), x, w, s, p)
  _x = permutedims(x, to_NHWC)
  _w = permutedims(w, [4, 3, 1, 2])
  _s = reverse(s)
  _out = jscall(:(math.conv2d), _x, _w, _s, p)
  permutedims(_out, to_NCHW)
end

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

function jscall(::typeof(maxpool), x, k, pad, stride)
  _x = permutedims(x, to_NHWC)
  _k = reverse(k)
  _s = reverse(stride)
  _out = jscall(:(math.maxPool), _x, _k, _s, pad)
  permutedims(_out, to_NCHW)
end

# broadcasted ops

# shape(::typeof(broadcast), f, xs...) =
#   Shape{eltype(xs[1])}(Base.Broadcast.broadcast_shape(size.(xs)...)...)

bcastable(+, *, tanh, relu, σ, -, /)

jscall(::typeof(broadcast), ::typeof(+), a, b) = jscall(:(math.add), a, b)
jscall(::typeof(broadcast), ::typeof(σ), x) = jscall(:(math.sigmoid), x)
jscall(::typeof(broadcast), ::typeof(tanh), x) = jscall(:(math.tanh), x)
jscall(::typeof(broadcast), ::typeof(relu), x) = jscall(:(math.relu), x)
jscall(::typeof(broadcast), ::typeof(*), x, y) = jscall(:(math.mul), y, x)
jscall(::typeof(broadcast), ::typeof(-), x, y) = jscall(:(math.sub), x, y)
jscall(::typeof(broadcast), ::typeof(/), x, y) = jscall(:(math.div), x, y)

# shape(::typeof(reshape), x::Shape{T}, i...) where T =
#   Shape{T}(Base._reshape_uncolon(x, i))
#
# shape(x) = x
# shape(x::Shape) = x
# shape(x::Tuple) = shape.(x)
# shape(x::AbstractArray) = Shape{eltype(x)}(size(x)...)
# shape(x::TrackedArray) = shape(x.data)

# reshape

@primitive Trace Base.reshape(parent, dims...) =
  ! any(x -> x isa StagedArray, (parent, dims...)) ?
  trace(reshape, parent, dims...) :
  begin
    dims = any(x -> val(x) isa Colon, dims) ?
    map((x, y) -> val(x) isa Colon? y : x
      , dims, Base._reshape_uncolon(val(parent), val.(dims))) : dims
    StagedArray(reshape, parent, dims)
  end

@primitive Trace function Base.reshape(parent, dims::StagedArray)
  @show parent, dims
  StagedArray(reshape, parent, dims)
end

jscall(::typeof(reshape), p, dims...) =
  jscall(:(math.reshape), p, jscall(tuple, reverse(dims)...))

jscall(::typeof(reshape), p, dims) =
  jscall(:(math.reshape), p, dims)

# size
@primitive Trace Base.size(x::StagedArray) =
  StagedArray(getindex, x, "shape", v=size(val(x)))

@primitive Trace Base.size(x, i) =
  ! any(x -> x isa StagedArray, (x, i)) ?
  trace(size, x, i) :
  begin
    _size = trace((x) -> size(x), x);
    index = trace((s, i)-> (length(s) - i), _size ,i) # js arrays are reversed
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

@primitive Trace Base.getindex(t::StagedArray, i::Int) =
  StagedArray(getindex, t, i - 1, v = val(t)[i])

@primitive Trace function Base.getindex(t, i::StagedArray{Int})
  index = overdub(Trace(), x -> x - 1, i)
  StagedArray(getindex, t, i, v = val(t)[val(i)])
end

@primitive Trace function tuple(args...)
  any(x -> x isa StagedArray, args) ? StagedArray(tuple, args...) : trace(tuple, args...)
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
jscall(::typeof(sub), x, y) = jscall(:(flux.sub), x, y)
jscall(::typeof(mul), x, y) = jscall(:(flux.mul), x, y)

@primitive Trace function (BN::BatchNorm)(x)
  μ, σ, γ, β, λ = BN.μ, BN.σ, BN.γ, BN.β, BN.λ

  dims = trace(x -> length(size(x)), x)
  channels = trace((x,dims) -> size(x, dims - 1), x, dims)

  affine_shape = trace((dims, channels) -> begin
    affine_shape = Flux.data(ones(Int, dims))
    affine_shape[dims-1] = channels # not traced
    affine_shape
  end, dims, channels)

  setShape = vertex(DataFlow.Lambda(1,
    vertex(DataFlow.Do(),
      vcall(setindex!, unwrap(affine_shape), unwrap(channels), 1), # index 2 of reversed array
      unwrap(affine_shape))
      ))

  affine_shape = wrap(affine_shape, vcall(setShape, x))

  trace((x, μ, σ, γ, β, λ, affine_shape) -> begin
    k = Tuple(affine_shape)
    μ = reshape(μ, k)
    σ = reshape(σ, k)
    γ = reshape(γ, k)
    β = reshape(β, k)
    λ.(γ .* ((x .- μ) ./ σ) .+ β)
  end, x, μ, σ, γ, β, λ, affine_shape)
end



@primitive Trace Base.length(s::StagedArray) = StagedArray(getindex, s, "length", v=length(val(s)))
@primitive Trace Base.ones(t, i::StagedArray) = StagedArray(ones, t, i)
@primitive Trace Tuple(t::StagedArray) = StagedArray(Flux.data, t, v=Tuple(val(t)))
@primitive Trace Flux.data(t::StagedArray) = StagedArray(Flux.data, t)

jscall(::typeof(Base.ones), t, i) = jscall(:(tf.ones), jscall(tuple, i), dtype(t))
jscall(::typeof(Flux.data), t) = jscall(:(flux.data), t)

dtype(::Type{Int}) = "int32"
dtype(::Type{Float32}) = "float32"

@primitive Trace Base.setindex!(A, X, i) =
  any(x -> x isa StagedArray , (A, X)) ?
  StagedArray(setindex!, A, X, trace((i) -> i - 1, i), v = setindex!(val(A), val(X), val(i))) :
  trace(setindex!, A, X, i)
