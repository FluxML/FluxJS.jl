using NNlib: cdims, padtuple, pdims, conv
using Vinyl: primitive

const to_NCHW = :([0, 3, 1, 2])
const to_NHWC = :([0, 2, 3, 1])

# Library of mathematical functions we consider primitive.
# TODO: store Julia functions and types, to avoid tensorflow.js-specific functions

matVecMul(args...) = *(args...)

@primitive Trace x::AbstractMatrix * y::AbstractMatrix =
  StagedArray(*, x, y)

jscall(::typeof(*), a, b) = jscall(:(math.matMul), b, a)

@primitive Trace x::AbstractMatrix * y::AbstractVector =
  StagedArray(matVecMul, x, y)

jscall(::typeof(matVecMul), a, b) = jscall(:(math.vectorTimesMatrix), b, a)

@primitive Trace x::AbstractArray + y::AbstractArray =
  StagedArray(+, x, y)

jscall(::typeof(+), a, b) = jscall(:(math.add), b, a)

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

# conv2d

@primitive Trace function (c::Conv)(x)
  out = conv(val(x), c.weight, stride = c.stride, pad = c.pad)
  pad = 0
  !all(x-> x == c.pad[1], c.pad)?
    throw(error("Assymetric padding is unsupported by deeplearn-js")):
    pad = c.pad[1]

  y = StagedArray(conv2d, stagedinputs(x)..., c.weight, padtuple(x,c.stride), pad, v=out)
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
bcastable(+, *, /, ^, tanh, σ, relu, leakyrelu, abs, exp, log, -, copy)
# copy for residual blocks

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
jscall(::typeof(broadcast), ::typeof(-), x, y) = jscall(:(math.sub), x, y)
jscall(::typeof(broadcast), ::typeof(copy), x) = jscall(:(flux.slice), x)

# reshape

@primitive Trace Base.reshape(parent, dims...) =
  ! any(x -> x isa StagedArray, (parent, dims...)) ?
  trace(reshape, parent, dims...) :
  begin
    dims = any(x -> val(x) isa Colon, dims) ? begin
      p = 1
      pos = 0
      for i=1:length(dims)
        !( val(dims[i]) isa Colon ) ?
         p = tracecall((p, v)-> p*v, p, dims[i]) : (pos = i)
      end
      c = tracecall((x, p) -> (count(x)/p), parent, p)
      (dims[1:pos-1]..., c, dims[pos+1:end]...)
    end : dims
    StagedArray(reshape, parent, dims..., v=reshape(val(parent), Int.(val.(dims))...))
  end

Base.count(x::AbstractArray) = prod(size(x))
@primitive Trace Base.count(x::AbstractArray) = StagedArray(getindex, x, :(String("size")), v=count(val(x)))

jscall(::typeof(reshape), p, dims...) =
  jscall(:(math.reshape), p, jscall(tuple, reverse(dims)...))

# size
@primitive Trace Base.size(x::StagedArray) =
  StagedArray(getindex, x, :(String("shape")), v=size(val(x)))

@primitive Trace Base.size(x, i) =
  ! any(x -> x isa StagedArray, (x, i)) ?
  trace(size, x, i) :
  begin
    index, _size = invertedindex(x, i)
    StagedArray(getindex, _size, index, v=size(val(x))[val(i)])
  end

# gate ( for LSTM and GRU )

@primitive ctx::Trace function Flux.gate(x::AbstractArray, h, n)
  out = Flux.gate(val(x), val(h), val(n))
  _start =  overdub(Trace(), (h, n) -> h * (n-1), h, n)
  StagedArray(view, x, _start, h, v=out)
end

jscall(::typeof(view), x, start, length) =
  jscall(:(math.slice), x, start, length)

@primitive Trace start(x::StagedArray) = StagedArray(start, x)
jscall(::typeof(start), x) = DataFlow.constant(0)

@primitive Trace Base.getfield(x, i) =
  ! any(x -> x isa StagedArray, (x, i)) ?
  trace(getfield, x, i) :
  StagedArray(getindex, x, "$i", v=getfield(val(x), val(i)))

@primitive Trace Base.getfield(x::StagedArray, i::Union{StagedArray{Int,0},Int}) =
  StagedArray(getindex, x, primitive(Trace(), -, i, 1), v=getfield(val(x), val(i)))

@primitive Trace Base.getfield(x, i::StagedArray{Int,0}) =
  StagedArray(getindex, x, primitive(Trace(), -, i, 1), v=getfield(val(x), val(i)))

@primitive Trace Base.indexed_next(x::StagedArray, i, state) =
  StagedArray(tuple, primitive(Trace(), getindex, x, i), primitive(Trace(), +, state, 1))

Base.getindex(x::StagedArray, i) = StagedArray(getindex, x, i - 1, v=getindex(val(x), i)) # for splat operator to work

@primitive Trace Base.getindex(t::StagedArray, i::Int) =
  StagedArray(getindex, t, i - 1, v = val(t)[i])

@primitive Trace function Base.getindex(t, i::StagedArray{Int,0})
  index = overdub(Trace(), x -> x - 1, i)
  StagedArray(getindex, t, i, v = val(t)[val(i)])
end

@primitive Trace tuple(args...) =
  any(x -> x isa StagedArray, args) ? StagedArray(tuple, args...) : trace(tuple, args...)

add(x, y) = x + y
sub(x, y) = x - y
mul(x, y) = x * y
div(x, y) = x / y

# for StagedArray{Int}
function binary_op(op, sub)
  @eval @primitive Trace ($op)(x::T, y::T) where {T<:StagedArray{<:Number,0}} = StagedArray($sub, x, y)
  @eval @primitive Trace ($op)(x::T, y::S) where {T<:StagedArray{S,0}} where {S<:Number} = StagedArray($sub, x, y)
  @eval @primitive Trace ($op)(x::S, y::T) where {T<:StagedArray{S,0}} where {S<:Number} = StagedArray($sub, x, y)
end

binary_op(+, add)
binary_op(*, mul)
binary_op(-, sub)
binary_op(/, div)

jscall(::typeof(add), x, y) = jscall(:(+), x, y)
jscall(::typeof(sub), x, y) = jscall(:(-), x, y)
jscall(::typeof(mul), x, y) = jscall(:(*), x, y)
jscall(::typeof(div), x, y) = jscall(:(/), x, y)

@primitive Trace function (BN::BatchNorm)(x)
  μ, σ, γ, β, λ = BN.μ, BN.σ, BN.γ, BN.β, BN.λ

  dims = trace(x -> length(size(x)), stagedinputs(x)...)
  channels = trace((x,dims) -> size(x, dims - 1), stagedinputs(x)..., dims)

  affine_shape = trace((dims, channels) -> begin
    affine_shape = onesArr(Int, dims)
    affine_shape[dims-1] = channels # not traced
    affine_shape
  end, dims, channels)

  dims_ = trace((dims)-> dims - 2, dims)

  out = trace((x) -> begin
    k = Tuple(affine_shape)
    μ = reshape(μ, affine_shape...)
    σ = reshape(σ, affine_shape...)
    γ = reshape(γ, affine_shape...)
    β = reshape(β, affine_shape...)
    λ.(γ .* ((x .- μ) ./ σ) .+ β)
  end, stagedinputs(x)...)

  f = vertex(DataFlow.Lambda(1,
    vertex(DataFlow.Do(),
      vcall(setindex!, unwrap(affine_shape), unwrap(channels), unwrap(dims_)),
      unwrap(out))
      ))

  wrap(out, vcall(f, x))
end

@primitive Trace Base.length(s::StagedArray) = StagedArray(getindex, s, :(String("length")), v=length(val(s)))
@primitive Trace Base.ones(t, i::StagedArray) = StagedArray(ones, t, i)
@primitive Trace Tuple(t::StagedArray) = StagedArray(Flux.data, t, v=Tuple(val(t)))
@primitive Trace Flux.data(t::StagedArray) = StagedArray(Flux.data, t)

jscall(::typeof(Base.ones), t, i) = jscall(:(tf.ones), jscall(tuple, i), dtype(t))
jscall(::typeof(Flux.data), t) = jscall(:(flux.data), t)

dtype(::Type{Int}) = :(String("int32"))
dtype(::Type{Float32}) = :(String("float32"))

onesArr(t, i) = ones(t, i)
@primitive Trace onesArr(t, i::StagedArray) = StagedArray(onesArr, t, i)
jscall(::typeof(onesArr), t, i) = jscall(:([].fill.apply), jscall(:(Array), i), :([1]))

@primitive Trace Base.setindex!(A, X, i) =
  any(x -> x isa StagedArray , (A, X)) ?
  StagedArray(setindex!, A, X, trace((i) -> i - 1, i), v = setindex!(val(A), val(X), val(i))) :
  trace(setindex!, A, X, i)

@primitive Trace copy(A::StagedArray{AbstractArray,N}) where N =
  StagedArray(copy,A)

jscall(::typeof(copy), A) = jscall(:(flux.slice), A)

@primitive Trace function mean(A::StagedArray, i)
  index, _ = invertedindex(A, i)
  StagedArray(mean, A, index, true, v=mean(val(A), val(i)))
end

jscall(::typeof(mean), x...) = jscall(:(math.mean), x...)

function invertedindex(x::AbstractArray, i)
  _size = trace((x) -> size(x), x)
  index = trace((s, i)-> (length(s) - i), _size ,i)
  return index, _size
end
