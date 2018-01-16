using Cassette
using Cassette: @context, @primitive, overdub
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
StagedArray{T,N}(f, args...) where {T,N} =
  StagedArray{T,N}(IVertex{Any}(f, graph.(args)...))

@context Trace

@primitive Trace (::typeof(*))(x::AbstractMatrix, y::AbstractVecOrMat) =
  StagedArray{Real,ndims(y)}(*, x, y)

stage(T::Type{<:AbstractArray}) = StagedArray{eltype(T),ndims(T)}
stage(T::Type{<:Real}) = StagedArray{T,0}

function trace(f, Ts...)
  inputs = [stage(T)(DataFlow.inputnode(n)) for (n, T) in enumerate(Ts)]
  overdub(Trace, f)(inputs...)
end

# f(a, b) = a*b
# trace(f, Array{Real,2}, Array{Real,1})
