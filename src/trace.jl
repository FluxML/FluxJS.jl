using Vinyl: @primitive, overdub, primitive, @hook
using DataFlow

struct Trace
  states::Vector{Any}
end

Trace() = Trace([])

struct StagedArray{T,N} <: AbstractArray{T,N}
  graph::IVertex{Any}
  val
end

Base.size(x::StagedArray) = size(val(x))

Base.show(io::IO, ::MIME"text/plain", s::StagedArray) =
  print(io, "StagedArray{$(eltype(s)),$(ndims(s))}($(s.val), $(s.graph))")

Base.show(io::IO, s::StagedArray) =
  print(io, "StagedArray{$(eltype(s)),$(ndims(s))}")

val(x) = x
val(x::StagedArray) = val(x.val)
val(x::Tuple) = val.(x)

dims(x) = length(x)
dims(x::AbstractArray) = ndims(x)
dims(x::StagedArray) = dims(val(x))

graph(x::StagedArray) = x.graph
graph(x) = DataFlow.constant(x)
vcall(args...) = DataFlow.vertex(DataFlow.Call(), graph.(args)...)

eval_(f, args) = val(f(val.(args)...))

StagedArray(f, args...; v=eval_(f, args)) =
  StagedArray{typeof(v),dims(v)}(vcall(f, args...),v)

stage(x::AbstractArray{T,N}, v) where {T,N} = StagedArray{T,N}(v, val(x))
stage(x::Tuple, v) = StagedArray{Tuple{eltype(x)},dims(x)}(v, val(x))
stage(x::Real, v) = StagedArray{typeof(x),dims(x)}(v, val(x))
stage(x, v) = error("Unsupported type $(typeof(x))")

trace(f, args...; meta = Trace()) = overdub(meta, f, args...)

unwrap(x::StagedArray) = x.graph
unwrap(xs::Tuple) = vcall(tuple, unwrap.(xs)...)
unwrap(x::Union{Number,AbstractArray{<:Number}}) = DataFlow.constant(x)

stagedinputs(xs...) = [stage(x, DataFlow.inputnode(n)) for (n, x) in enumerate(xs)]

function _traceλ(f, args...; meta = Trace())
  out = trace(f, stagedinputs(args...)..., meta = meta)
  v = unwrap(out)
  out, vertex(DataFlow.Lambda(length(args), v))
end

traceλ(f, args...; meta = Trace()) = _traceλ(f, args..., meta = meta)[2]

wrap(x::StagedArray, v) = stage(x, v)
wrap(x::Tuple, v) = ntuple(n -> wrap(x[n], vertex(DataFlow.Split(n), v)), length(x))

function tracecall(f, args...; meta = Trace())
  out, v = _traceλ(f, args..., meta = meta)
  v = vcall(v, args...)
  wrap(out, v)
end

@primitive ctx::Trace function (f::Any)(args...)
  inline = !(all(x -> x isa Union{AbstractArray,Number}, args) &&
             any(x -> x isa StagedArray, args))
  inline ?
    trace(f, args..., meta = ctx) :
    tracecall(f, args..., meta = ctx)
end

# struct Bag{N} <: AbstractArray{Int,N}
#   arg
#   ctx
# end
#
# Base.size(b::Bag) = size(b.arg)
#
# (f::Dense)(b::Bag) = trace(f, b.arg, meta = b.ctx)
#
# @primitive ctx::Trace function (f::typeof(invoke))(foo::Dense, argtypes, arg::StagedArray{T,N}) where {T,N}
#   trace(foo, Bag{N}(arg, ctx), meta = ctx)
# end

control(a::IVertex, b::IVertex = DataFlow.inputnode()) = vcall(control, a, b)

@primitive ctx::Trace function (f::Flux.Recur)(args...)
  push!(ctx.states, f.init)
  i = length(ctx.states)-1
  states = control(DataFlow.constant(:states))
  state = stage(f.init, vcall(getindex, states, i))
  h, y = trace(f.cell, state, stagedinputs(args...)...)
  λ = vertex(DataFlow.Lambda(length(args),
                             vertex(DataFlow.Do(),
                                    vcall(setindex!, states, unwrap(h), i),
                                    graph(y))))
  wrap(y, vcall(λ, args...))
end

# Base.Broadcast.broadcasted(::Type{T}, x::Type{StagedArray{T,N}}) where {T,N} = x
# Base.Broadcast.broadcasted(::Type{T}, x::Type{StagedArray{S,N}}) where {T,S,N} =

# @primitive BTrace (::Type{T})(arg) where T =
#   isa(arg, Type{StagedArray}) ?


struct BTrace end
@primitive Trace function (::typeof(Base.Broadcast.broadcasted))(f, args...)
  overdub(BTrace(), () -> f(args...))
end

# don't record type conversions
@primitive Trace function (::typeof(Base.Broadcast.broadcasted))(f::Type{T}, x::StagedArray) where {T<:Number}
  v = val(f).(val(x))
  StagedArray{eltype(v),dims(v)}(graph(x), v)
end

# @primitive Trace function (::typeof(Base.Broadcast.broadcasted))(f::Type{T}, arg::StagedArray{F,N}) where {T, F, N}
#   println("yup $f $arg")
#   StagedArray(graph(arg), val(f).(val(arg)))
# end
# @primitive Trace (::typeof(Base.Broadcast.materialize))(bc) = bc
@primitive Trace (::typeof(broadcast))(f, args...) = overdub(BTrace(), () -> f(args...))

bcast_ndims(args...) = maximum(arg isa AbstractArray ? ndims(arg) : 0 for arg in args)

function bcastable(op)
  @eval @primitive BTrace (::typeof($op))(args...) = StagedArray(broadcast, $op, args...)
end

bcastable(op, ops...) = (bcastable(op); bcastable(ops...))
