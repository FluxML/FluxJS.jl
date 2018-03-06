using Flux, FluxJS, DataFlow, MacroTools
using FluxJS: traceλ
using Base.Test

@testset "FluxJS" begin

m = Dense(10,5)
v = traceλ(m, rand(10))
ex = prettify(DataFlow.syntax(traceλ(m,rand(10))))

@test @capture ex _ -> (+).(matVecMul(_,_),_)

end
