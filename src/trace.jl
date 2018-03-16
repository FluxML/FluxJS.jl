using Vinyl: @primitive, overdub
using DataFlow
import Flux.JIT: shape

struct Trace
  states::Vector{Any}
end

Trace() = Trace([])

struct StagedArray{T,N} <: AbstractArray{T,N}
  graph::IVertex{Any}
  dims::NTuple{N,Int}
end

Base.size(x::StagedArray) = x.dims

Base.show(io::IO, ::MIME"text/plain", s::StagedArray) =
  print(io, "StagedArray{$(eltype(s))}($(s.dims), $(s.graph))")

Base.show(io::IO, s::StagedArray) =
  print(io, "StagedArray{$(eltype(s))}($(s.dims))")

graph(x::StagedArray) = x.graph
graph(x) = DataFlow.constant(x)
vcall(args...) = DataFlow.vertex(DataFlow.Call(), graph.(args)...)

function StagedArray(f, args...)
  sh = shape(f, shape.(args)...)
  StagedArray{eltype(sh),ndims(sh)}(vcall(f, args...),sh.dims)
end

stage(x::AbstractArray{T,N}, v) where {T,N} = StagedArray{T,N}(v, size(x))
stage(x::Real, v) = StagedArray{typeof(x),0}(v, ())
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
  h, y = trace(f.cell, state, stagedinputs(args...)...)
  λ = vertex(DataFlow.Lambda(length(args),
                             vertex(DataFlow.Do(),
                                    vcall(setindex!, states, unwrap(h), i),
                                    graph(y))))
  wrap(y, vcall(λ, args...))
end

struct BTrace end
@primitive Trace (::typeof(broadcast))(f, args...) = overdub(BTrace(), () -> f(args...))

bcast_ndims(args...) = maximum(arg isa AbstractArray ? ndims(arg) : 0 for arg in args)

function bcastable(op)
  @eval @primitive BTrace (::typeof($op))(args...) = StagedArray(broadcast, $op, args...)
end

bcastable(op, ops...) = (bcastable(op); bcastable(ops...))
