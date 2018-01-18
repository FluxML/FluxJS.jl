blob_version = UInt8(1)

PRIMITIVES = [Float16, Float32, Float64, Int16, Int32, Int64]

TAGS = [Any, Vector, Array, PRIMITIVES...]

for T in [Any, PRIMITIVES...]
  @eval tag(io::IO, ::Type{$T}) = write(io, $(UInt8(findfirst(TAGS, T))))
end

for T in [Float16, Float32, Float64, Int16, Int32, Int64]
  @eval data(io::IO, x::$T) = write(io, x)
end

function tag(io::IO, x::Type{<:AbstractVector{T}}) where T
  write(io, UInt8(findfirst(TAGS, Vector)))
  tag(io, T in PRIMITIVES ? T : Any)
end

function data(io::IO, x::AbstractVector)
  write(io, UInt32(length(x)))
  if eltype(x) in PRIMITIVES
    write(io, x)
  else
    foreach(x -> _blob(io, x), x)
  end
end

function _blob(io::IO, x)
  tag(io, typeof(x))
  data(io, x)
end

function blob(io::IO, x)
  write(io, UInt8(blob_version))
  _blob(io, x)
end

# open(io -> blob(io, Any[[1.5f0, 3.5f0], [2.5, 4.5]]), "lib/test.blob", "w")
# read("lib/test.blob")
