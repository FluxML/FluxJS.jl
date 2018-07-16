using Flux, FluxJS, DataFlow, MacroTools
using FluxJS: traceλ
using Base.Test
using JSExpr, JSON, WebIO
using Blink

include("./helpers.jl")

@testset "FluxJS" begin

m = Dense(10,5)
v = traceλ(m, rand(10))
ex = prettify(DataFlow.syntax(traceλ(m,rand(10))))

@test @capture ex _ -> (+).(matVecMul(_,_),_)

    @testset "Dense Layer" begin
        m = Chain(Dense(10, 10))
        x = rand(10)
        testjs(m, x)
    end

    @testset "broadcast" begin
        m = Chain(x -> x .+ ones(10))
        x = rand(10)
        testjs(m, x)
    end

end
