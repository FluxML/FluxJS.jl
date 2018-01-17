module FluxJS

using Flux, MacroTools

include("trace.jl")
include("lib.jl")
include("compile.jl")
include("dump.jl")

end # module
