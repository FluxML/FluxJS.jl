using Cassette
using Cassette: @context, @primitive
using DataFlow

struct StagedArray{T,N} <: AbstractArray{T,N}
  graph::IVertex{Any}
end

Base.show(io::IO, ::MIME"text/plain", s::StagedArray) =
  print(io, "StagedArray{$(eltype(s)),$(ndims(s))}($(s.graph))")

Base.show(io::IO, s::StagedArray) =
  print(io, "StagedArray{$(eltype(s)),$(ndims(s))}()")

graph(x::StagedArray) = x.graph
graph(x) = DataFlow.constant(x)
graph(x::Flux.TrackedArray) = graph(Flux.Tracker.value(x))
vcall(args...) = DataFlow.vertex(DataFlow.Call(), graph.(args)...)
StagedArray{T,N}(f, args...) where {T,N} = StagedArray{T,N}(vcall(f, args...))

Cassette.@context Trace

stage(T::Type{<:AbstractArray}) = StagedArray{eltype(T),ndims(T)}
stage(T::Type{<:Real}) = StagedArray{T,0}

@primitive Trace (f::Core.Builtin)(args...) = f(args...)

@primitive Trace function (f::Any)(args...)
  (all(x -> x isa Union{AbstractArray,Number}, args) &&
    any(x -> x isa StagedArray, args)) || return exec(f, args...)
  T, v = _trace(f, typeof.(args)...)
  T <: StagedArray ? T(v, args...) : v
end

# Trace through broadcast calls
Cassette.@context BTrace
@primitive Trace (::typeof(broadcast))(f, args...) = Cassette.overdub(BTrace, f)(args...)

bcast_ndims(args...) = maximum(arg isa AbstractArray ? ndims(arg) : 0 for arg in args)

function bcastable(op)
  @eval @primitive BTrace (::typeof($op))(args...) =
    StagedArray{Real,bcast_ndims(args...)}(vcall(broadcast, $op, args...))
end

bcastable(op, ops...) = (bcastable(op); bcastable(ops...))

lambda(v, args) = vertex(DataFlow.Lambda(args, v))

exec(f, args...) = Cassette.execute(Val(false), Cassette.overdub(Trace, f), args...)

function _trace(f, Ts...)
  inputs = [stage(T)(DataFlow.inputnode(n)) for (n, T) in enumerate(Ts)]
  a = exec(f, inputs...)
  typeof(a), lambda(graph(a), length(Ts))
end

trace(f, Ts...) = _trace(f, Ts...)[2]
