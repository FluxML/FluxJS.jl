module FluxJS

using Flux, MacroTools, BSON

export @code_js

include("dump.jl")
include("trace.jl")
include("compile.jl")
include("lib.jl")

end # module
