using NNlib: conv, DenseConvDims, PoolDims
using Flux: maxpool

const to_NCHW = :([0, 3, 1, 2])
const to_NHWC = :([0, 2, 3, 1])

# Library of mathematical functions we consider primitive.
# TODO: store Julia functions and types, to avoid tensorflow.js-specific functions

matVecMul(args...) = *(args...)

@primitive Trace x::AbstractMatrix * y::AbstractMatrix =
  Staged(*, x, y)

jscall(::typeof(*), a, b) = jscall(:(math.matMul), b, a)

@primitive Trace x::AbstractMatrix * y::AbstractVector =
  Staged(matVecMul, x, y)

jscall(::typeof(matVecMul), a, b) = jscall(:(math.vectorTimesMatrix), b, a)

@primitive Trace x::AbstractArray + y::AbstractArray =
  Staged(+, x, y)

jscall(::typeof(+), a, b) = jscall(:(math.add), b, a)

# cat

concat1D(a, b) = vcat(a, b)
concat(a, b) = vcat(a, b)

@primitive Trace vcat(a::AbstractVector, b::AbstractVector) =
  Staged(concat1D, a, b)

@primitive Trace vcat(a::AbstractMatrix, b::AbstractMatrix) =
  Staged(concat, a, b)

jscall(::typeof(concat1D), a, b) = jscall(:(math.concat1d), jscall(tuple,a, b))
jscall(::typeof(concat), a, b) = jscall(:(math.concat), jscall(tuple, a, b), -1)

# softmax

@primitive Trace softmax(x::AbstractVecOrMat) =
  Staged(softmax, x)

jscall(::typeof(softmax), x) = jscall(:(math.softmax), x)

# Dense uses invoke which JuliaInterpreter does not enter
@primitive Trace function (a::Dense)(x::AbstractArray)
  W, b, σ = a.W, a.b, a.σ
  trace(() -> σ.(W*x .+ b))
end

# conv2d

@primitive Trace function (c::Conv)(x)
  cdims = DenseConvDims(val(x), c.weight; stride=c.stride, padding=c.pad, dilation=c.dilation) 
  out = conv(val(x), c.weight, cdims)

  all(x-> x == c.pad[1], c.pad) ||
    error("Assymetric padding is unsupported by tf.conv2d")

  y = Staged(conv, stagedinputs(x)..., c.weight, c.stride, c.pad[1], c.dilation, v=out)
  σ, b = c.σ, reshape(c.bias, map(_->1, c.stride)..., :, 1)
  out = overdub(Trace(), (x) -> (σ).(x .+ b), y)
  wrap(out, vcall(vertex(DataFlow.Lambda(1, unwrap(out))), x))
end

Base.permutedims(x::Union{Staged,IVertex}, p) = jscall(:(math.transpose), x, p)
Base.reverse(x::Staged) = jscall(:(math.transpose), x)

# tf-js uses NHWC while js default is NCHW
function jscall(::typeof(conv), x, w, s, p, d)
  _x = permutedims(x, to_NHWC)
  _w = jscall(:(math.reverse), jscall(:(math.transpose), w, :([2, 3, 1,0]), x), :([0,1]))
  _s = reverse(s)
  _d = reverse(d)
  _out = jscall(:(math.conv2d), _x, _w, _s, p, "NHWC", _d, "floor")
  permutedims(_out, to_NCHW)
end

# maxpool

@primitive Trace function (m::MaxPool)(x)
  pdims = PoolDims(val(x), m.k; padding=m.pad, stride=m.stride)
  out = maxpool(val(x), pdims)

  all(x-> x == m.pad[1], m.pad) ||
    throw(error("Assymetric padding is unsupported by deeplearn-js"))

  Staged(maxpool, x, m.k, m.pad[1], m.stride, v=out)
end

function jscall(::typeof(maxpool), x, k, pad, stride)
  _x = permutedims(x, to_NHWC)
  _k = reverse(k)
  _s = reverse(stride)
  _out = jscall(:(math.maxPool), _x, _k, _s, pad, "floor")
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

Base.reshape(A::AbstractArray, i...) = nothing

@primitive Trace Base.reshape(parent::AbstractArray, dims::Union{Colon,StagedReal}...) =
  ! any(x -> x isa StagedType, (parent, dims...)) ?
  trace(reshape, parent, dims...) :
  begin
    dims = any(x -> val(x) isa Colon, dims) ? begin
      # resolve dimensions
      p = 1
      pos = 0
      for i=1:length(dims)
        !( val(dims[i]) isa Colon ) ?
         p = tracecall((p, v)-> p*v, p, dims[i]) : (pos = i)
      end
      c = tracecall((x, p) -> (count(x)/p), parent, p)
      (dims[1:pos-1]..., c, dims[pos+1:end]...)
    end : dims
    Staged(reshape, parent, dims..., v=reshape(val(parent), Int.(val.(dims))...))
  end

Base.count(x::AbstractArray) = prod(size(x))
@primitive Trace Base.count(x::AbstractArray) = Staged(getindex, x, :(String("size")), v=count(val(x)))

jscall(::typeof(reshape), p, dims...) =
  jscall(:(math.reshape), p, jscall(tuple, reverse(dims)...))

# size
@primitive Trace Base.size(x::StagedArray) =
  Staged(getindex, x, :("shape"), v=size(val(x)))

@primitive ctx::Trace Base.size(x, i) =
  !any(x -> x isa StagedType, (x, i)) ?
  trace(size, x, i) :
  begin
    s, n = invertedindex(x, i)
    wrap(s[i], unwrap(s[n]))
  end

# # gate ( for LSTM and GRU )
@primitive ctx::Trace function Flux.gate(x::AbstractArray, h, n)
  out = Flux.gate(val(x), val(h), val(n))
  _start =  trace((h, n) -> h * (n-1), h, n)
  Staged(view, x, _start, h, v=out)
end

jscall(::typeof(view), x, start, length) =
  jscall(:(math.slice), x, start, length)

add(x, y) = x + y
sub(x, y) = x - y
mul(x, y) = x * y
div(x, y) = x / y

# for Staged{Int}
function binary_op(op, sub)
  @eval @primitive Trace ($op)(x::Number, y::Number) =
    any(e-> e isa StagedReal, (x, y)) ? Staged($sub, x, y) : $(op)(x, y)
end

binary_op(+, add)
binary_op(*, mul)
binary_op(-, sub)
binary_op(/, div)

jscall(::typeof(add), x, y) = jscall(:(+), x, y)
jscall(::typeof(sub), x, y) = jscall(:(-), x, y)
jscall(::typeof(mul), x, y) = jscall(:(*), x, y)
jscall(::typeof(div), x, y) = jscall(:(/), x, y)

bn(args...) = nothing

@primitive Trace function (BN::BatchNorm)(x)
  μ, σ², ϵ, γ, β, λ = BN.μ, BN.σ², BN.ϵ, BN.γ, BN.β, BN.λ
  s = Staged(bn, stagedinputs(x)..., μ, σ², ϵ, γ, β, v=BN(val(x)))
  y = trace((l) -> λ.(l), s)
end

jscall(::typeof(bn), x, μ, σ², ϵ, γ, β) =
  jscall(:(tf.batchNormalization), x, μ, σ², ϵ, γ, β)

@primitive Trace copy(A::Staged{T}) where {T <: AbstractArray} =
  Staged(copy,A)

jscall(::typeof(copy), A) = jscall(:(flux.slice), A)

function invertedindex(x::AbstractArray, i)
  _size = trace((x) -> size(x), x)
  l = length(val(_size))
  index = trace((l, i)-> l - i + 1, l, i)
  return _size, index
end
