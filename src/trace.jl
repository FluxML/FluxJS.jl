using Vinyl: @primitive, overdub, isprimitive, primitive
using DataFlow
import ASTInterpreter2._Typeof

struct Trace
  states::Vector{Any}
end

Trace() = Trace([])

struct StagedArray{T,N} <: AbstractArray{T,N}
  graph::IVertex{Any}
  val
end

Base.size(x::StagedArray) = size(x.val)

Base.show(io::IO, ::MIME"text/plain", s::StagedArray) =
  print(io, "StagedArray{$(eltype(s)),$(ndims(s))}($(s.val), $(s.graph))")

Base.show(io::IO, s::StagedArray) =
  print(io, "StagedArray{$(eltype(s)),$(ndims(s))}")

val(x) = x
val(x::StagedArray) = val(x.val)
val(x::Tuple) = val.(x)

dims(x) = ndims(x)
dims(x::Tuple) = length(x)
dims(x::StagedArray) = dims(val(x))

_Typeof(x) = isa(x,Type) ? Type{x} : typeof(val(x))

graph(x::StagedArray) = x.graph
graph(x) = DataFlow.constant(x)
vcall(args...) = DataFlow.vertex(DataFlow.Call(), graph.(args)...)

function StagedArray(f, args...; v=val(f(val.(args)...)))
  @show f, args, v
  StagedArray{typeof(v),dims(v)}(vcall(f, args...),v)
end

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

control(a::IVertex, b::IVertex = DataFlow.inputnode()) = vcall(control, a, b)

@primitive ctx::Trace function (f::Flux.Recur)(args...)
  push!(ctx.states, f.init)
  i = length(ctx.states)-1
  states = control(DataFlow.constant(:states))
  state = stage(f.init, vcall(getindex, states, i))
  out = trace(f.cell, state, stagedinputs(args...)...)
  h = trace((o) -> o[1], out)
  y = trace((o) -> o[2], out)
  λ = vertex(DataFlow.Lambda(length(args),
                             vertex(DataFlow.Do(),
                                    vcall(setindex!, states, unwrap(h), i),
                                    graph(y))))
  @show λ
  wrap(y, vcall(λ, args...))
end

struct BTrace end
@primitive Trace (::typeof(broadcast))(f, args...) = overdub(BTrace(), () -> f(args...))

bcast_ndims(args...) = maximum(arg isa AbstractArray ? ndims(arg) : 0 for arg in args)

function bcastable(op)
  @eval @primitive BTrace (::typeof($op))(args...) = StagedArray(broadcast, $op, args...)
end

bcastable(op, ops...) = (bcastable(op); bcastable(ops...))
