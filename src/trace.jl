using Vinyl: @primitive, overdub, primitive, @hook
using DataFlow
using JuliaInterpreter

struct Trace
  states::Vector{Any}
end

Trace() = Trace([])

struct Staged{T}
  graph::IVertex{Any}
  val::T
end

struct StagedArray{T,N,R<:AbstractArray{T,N}} <: AbstractArray{T,N}
  s::Staged{R}
end

struct StagedReal{N <: Real} <: Real
  s::Staged{N}
end

const StagedType = Union{StagedArray,StagedReal}

Base.show(io::IO, s::Union{StagedType,Staged}) =
  print(io, "$(typeof(s))")

Base.show(io::IO, ::MIME"text/plain", s::Union{StagedType,Staged}) =
  print(io, "$(typeof(s))($(unwrap(s)), $(val(s)))")

val(x) = x
val(x::Staged) = val(x.val)
val(x::StagedType) = val(x.s)
val(x::Tuple) = val.(x)

graph(x::Staged) = x.graph
graph(x::StagedType) = graph(x.s)
graph(x) = DataFlow.constant(x)
vcall(args...) = DataFlow.vertex(DataFlow.Call(), graph.(args)...)

eval_(f, args) = val(f(val.(args)...))

Staged(f, args...; v=eval_(f, args)) = stage(v, vcall(f, args...))

stage(x::AbstractArray{T,N}, v) where {T,N} = StagedArray(Staged(v, val(x)))
stage(x::StagedType, v) = stage(val(x), v)
stage(x::Tuple, v) = ntuple(n -> stage(x[n], vcall(getindex, v, n - 1)), length(x))
stage(x::Real, v) = StagedReal(Staged(v, val(x)))
stage(x, v) = error("Unsupported type $(typeof(x))")

trace(f, args...; meta = Trace()) = overdub(meta, f, args...)

unwrap(x::Staged) = x.graph
unwrap(xs::Tuple) = vcall(tuple, unwrap.(xs)...)
unwrap(x::Union{Number,AbstractArray{<:Number}}) = DataFlow.constant(x)
unwrap(x::StagedType) = unwrap(x.s)

stagedinputs(xs...) = [stage(x, DataFlow.inputnode(n)) for (n, x) in enumerate(xs)]

function _traceλ(f, args...; meta = Trace())
  out = trace(f, stagedinputs(args...)..., meta = meta)
  v = unwrap(out)
  out, vertex(DataFlow.Lambda(length(args), v))
end

traceλ(f, args...; meta = Trace()) = _traceλ(f, args..., meta = meta)[2]

wrap(args...) = stage(args...)

function tracecall(f, args...; meta = Trace())
  out, v = _traceλ(f, args..., meta = meta)
  v = vcall(v, args...)
  wrap(out, v)
end

iscompiled(f) = parentmodule(typeof(f)) == JuliaInterpreter.CompiledCalls

@primitive ctx::Trace function (f::Any)(args...)
  iscompiled(f) && return Base.invokelatest(f, args...)
  inline = !(all(x -> x isa Union{AbstractArray,Number}, args) &&
             any(x -> x isa StagedType, args))
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
@primitive Trace function (::typeof(Base.Broadcast.broadcasted))(f, args...)
  overdub(BTrace(), () -> f(args...))
end

# don't record type conversions
@primitive Trace function (::typeof(Base.Broadcast.broadcasted))(f::Type{T}, x::StagedType) where {T<:Number}
  stage(f.(val(x)), unwrap(x))
end

@primitive Trace (::typeof(Base.Broadcast.materialize))(bc) = bc
@primitive Trace (::typeof(broadcast))(f, args...) = overdub(BTrace(), () -> f(args...))

bcast_ndims(args...) = maximum(arg isa AbstractArray ? ndims(arg) : 0 for arg in args)

function bcastable(op)
  @eval @primitive BTrace (::typeof($op))(args...) = Staged(broadcast, $op, args...)
end

bcastable(op, ops...) = (bcastable(op); bcastable(ops...))
