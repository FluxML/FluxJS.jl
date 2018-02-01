using Cassette
using Cassette: @context, @primitive
using DataFlow

struct TraceCtx
  states::Vector{Any}
end

TraceCtx() = TraceCtx([])

struct StagedArray{T,N} <: AbstractArray{T,N}
  graph::IVertex{Any}
end

Base.show(io::IO, ::MIME"text/plain", s::StagedArray) =
  print(io, "StagedArray{$(eltype(s)),$(ndims(s))}($(s.graph))")

Base.show(io::IO, s::StagedArray) =
  print(io, "StagedArray{$(eltype(s)),$(ndims(s))}()")

graph(x::StagedArray) = x.graph
graph(x) = DataFlow.constant(x)
vcall(args...) = DataFlow.vertex(DataFlow.Call(), graph.(args)...)
StagedArray{T,N}(f, args...) where {T,N} = StagedArray{T,N}(vcall(f, args...))

stage(T::Type{<:AbstractArray}) = StagedArray{eltype(T),ndims(T)}
stage(T::Type{<:Real}) = StagedArray{T,0}
stage(T::Type) = error("Unsupported type $T")
stage(x) = stage(typeof(x))

Cassette.@context Trace

trace(f, args...; meta = TraceCtx()) =
  Cassette.execute(Val(false), Cassette.overdub(Trace, f, metadata = meta), args...)

unwrap(x::StagedArray) = x.graph
unwrap(xs::Tuple) = vcall(tuple, unwrap.(xs)...)

stagedinputs(Ts...) = [stage(T)(DataFlow.inputnode(n)) for (n, T) in enumerate(Ts)]

function _traceλ(f, args...; meta = TraceCtx())
  out = trace(f, stagedinputs(args...)..., meta = meta)
  v = unwrap(out)
  out, vertex(DataFlow.Lambda(length(args), v))
end

traceλ(f, args...; meta = TraceCtx()) = _traceλ(f, args..., meta = meta)[2]

wrap(x::StagedArray, v) = typeof(x)(v)
wrap(x::Tuple, v) = ntuple(n -> wrap(x[n], vertex(DataFlow.Split(n), v)), length(x))

function tracecall(f, args...; meta = TraceCtx())
  out, v = _traceλ(f, args..., meta = meta)
  v = vcall(v, args...)
  wrap(out, v)
end

# Avoid stack overflow
@primitive Trace (f::Core.Builtin)(args...) = f(args...)

@primitive Trace ctx function (f::Any)(args...)
  # Avoid stack overflow
  applicable(f, args...) || return f(args...)
  (all(x -> x isa Union{AbstractArray,Number}, args) &&
    any(x -> x isa StagedArray, args)) || return trace(f, args..., meta = ctx)
  tracecall(f, args...)
end

control(a::IVertex, b::IVertex = DataFlow.inputnode()) = vcall(control, a, b)

@primitive Trace ctx function (f::Flux.Recur)(args...)
  push!(ctx.states, f.init)
  i = length(ctx.states)
  vstate = control(DataFlow.constant(:state))
  h, y = trace(f.cell, stage(f.init)(getindex, vstate, i), args...)
  typeof(y)( # TODO: wrap properly
    vertex(DataFlow.Do(),
      vcall(setindex!, vstate, unwrap(h), i),
        graph(y)))
end

Cassette.@context BTrace
@primitive Trace (::typeof(broadcast))(f, args...) = Cassette.overdub(BTrace, f)(args...)

bcast_ndims(args...) = maximum(arg isa AbstractArray ? ndims(arg) : 0 for arg in args)

function bcastable(op)
  @eval @primitive BTrace (::typeof($op))(args...) =
    StagedArray{Real,bcast_ndims(args...)}(vcall(broadcast, $op, args...))
end

bcastable(op, ops...) = (bcastable(op); bcastable(ops...))
