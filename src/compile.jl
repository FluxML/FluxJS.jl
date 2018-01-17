using DataFlow
using DataFlow: vcall

jsarray_constructor(x) = :(dl.$(Symbol("Array$(ndims(x))D")).new)
jsarray(x::AbstractVector) = vcall(jsarray_constructor(x), collect(reshape(x, :)))
jsarray(x) = vcall(jsarray_constructor(x), size(x), collect(reshape(x, :)))

jscall(args...) = vcall(args...)

jscall(::typeof(*), a, b) = jscall(:(math.matMul), a, b)
jscall(::typeof(matVecMul), a, b) = jscall(:(math.matrixTimesVector), a, b)
jscall(::typeof(broadcast), ::typeof(+), a, b) = jscall(:(math.add), a, b)
jscall(::typeof(broadcast), ::typeof(σ), x) = jscall(:(math.sigmoid), x)
jscall(::typeof(softmax), x) = jscall(:(math.softmax), x)

cvalue(v) = DataFlow.isconstant(v) ? v.value.value : v

function lower(v)
  DataFlow.postwalk(DataFlow.λopen(model)) do v
    cvalue(v) isa Array && return jsarray(cvalue(v))
    DataFlow.iscall(v) || return v
    jscall(cvalue.(v[:])...)
  end |> DataFlow.λclose
end
