module FluxJS

using Flux, MacroTools

export @code_js

include("trace.jl")
include("compile.jl")
include("lib.jl")
include("dump.jl")
include("blob.jl")

end # module
