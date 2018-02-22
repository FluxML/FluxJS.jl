module BSON

@enum(BSONType::UInt8,
  double=1, string, document, array, binary, undefined, objectid, boolean,
  datetime, null, regex, dbpointer, javascript, symbol, javascript_scoped,
  int32, timestamp, int64, decimal128, minkey=0xFF, maxkey=0x7F)

bsontype(::Int64) = int64
bsontype(::Int32) = int32
bsontype(::Float64) = double
bsontype(::String) = string
bsontype(::Vector{UInt8}) = binary
bsontype(::Associative) = document

bsonsize(x) = sizeof(x)
bsonsize(x::String) = sizeof(x)+5
bsonsize(x::Vector{UInt8}) = sizeof(x)+5
bsonsize(x::Associative) = sum(1+sizeof(k)+1+bsonsize(v) for (k,v) in x)+5

lower(x::Int32) = x
lower(x::Integer) = Int64(x)
lower(x::Real) = Float64(x)
lower(x::String) = x
lower(x::Vector{UInt8}) = x

lower(doc::Associative) = Dict(Base.string(k) => lower(v) for (k, v) in doc)

_bson(io::IO, x) = write(io, x)

function _bson(io::IO, x::String)
  write(io, Int32(sizeof(x)+1))
  write(io, x, 0x00)
end

function _bson(io::IO, buf::Vector{UInt8})
  write(io, Int32(length(buf)))
  write(io, 0x00)
  write(io, buf)
end

function _bson(io::IO, doc::Associative)
  write(io, Int32(bsonsize(doc)))
  for (k, v) in doc
    write(io, bsontype(v))
    write(io, k::String, 0x00)
    _bson(io, v)
  end
  write(io, 0x00)
end

function bson(io::IO, doc::Associative)
  doc = lower(doc)
  _bson(io, doc)
end

bson(p::AbstractString, x) = open(io -> bson(io, x), p, "w")

# Lists

struct List
  dict::Dict{String,Any}
end

lower(xs::Vector) = List(lower(Dict(i => x for (i,x) in enumerate(xs))))
bsontype(xs::List) = array
bsonsize(xs::List) = bsonsize(xs.dict)
_bson(io::IO, xs::List) = _bson(io, xs.dict)

# Arrays

lower(xs::Array{T}) where T <: Real =
  Dict(:dims => Any[UInt32.(size(xs))...],
       :type => lowercase(Base.string(eltype(xs))),
       :data => reinterpret(UInt8, reshape(xs, :))) |> lower

end
