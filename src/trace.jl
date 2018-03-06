using Vinyl: @primitive, overdub
using DataFlow

struct Trace
  states::Vector{Any}
end

Trace() = Trace([])

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

trace(f, args...; meta = Trace()) = overdub(meta, f, args...)

unwrap(x::StagedArray) = x.graph
unwrap(xs::Tuple) = vcall(tuple, unwrap.(xs)...)
unwrap(x::Union{Number,AbstractArray{<:Number}}) = DataFlow.constant(x)

stagedinputs(Ts...) = [stage(T)(DataFlow.inputnode(n)) for (n, T) in enumerate(Ts)]

function _traceλ(f, args...; meta = Trace())
  out = trace(f, stagedinputs(args...)..., meta = meta)
  v = unwrap(out)
  out, vertex(DataFlow.Lambda(length(args), v))
end

traceλ(f, args...; meta = Trace()) = _traceλ(f, args..., meta = meta)[2]

wrap(x::StagedArray, v) = typeof(x)(v)
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
  @show f
  push!(ctx.states, f.init)
  i = length(ctx.states)-1
  vstate = control(DataFlow.constant(:states))
  h, y = trace(f.cell, stage(f.init)(getindex, vstate, i), stagedinputs(args...)...)
  λ = vertex(DataFlow.Lambda(length(args),
                             vertex(DataFlow.Do(),
                                    vcall(setindex!, vstate, unwrap(h), i),
                                    graph(y))))
  typeof(y)(vcall(λ, args...)) # TODO: wrap properly
end

struct BTrace end
@primitive Trace (::typeof(broadcast))(f, args...) = overdub(BTrace(), () -> f(args...))

bcast_ndims(args...) = maximum(arg isa AbstractArray ? ndims(arg) : 0 for arg in args)

function bcastable(op)
  @eval @primitive BTrace (::typeof($op))(args...) =
    StagedArray{Real,bcast_ndims(args...)}(vcall(broadcast, $op, args...))
end

bcastable(op, ops...) = (bcastable(op); bcastable(ops...))
