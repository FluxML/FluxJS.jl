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

w = setupWindow()

x = rand(2)
@testset "*" begin
    m = (x) -> ones(2, 2) * x
    testjs(w, m, x)
    testjs(w, m, rand(2, 1))
end

@testset "+" begin
    m = (x) -> x + ones(2)
    testjs(w, m, x)
end

@testset "vcat" begin
    m = (x) -> vcat(x, ones(2))
    testjs(w, m, x)

    m = (x) -> vcat(x, ones(2, 2))
    testjs(w, m, rand(3, 2))
end

@testset "softmax" begin
    m = x -> softmax(x)
    testjs(w, m, x)
end

@testset "Dense Layer" begin
    m = Dense(2, 2)
    testjs(w, m, x)
end

@testset "broadcast" begin
    b = ones(2)

    m = Chain(x -> x .+ b)
    testjs(w, m, x)

    m = Chain(x -> x .* b)
    testjs(w, m, x)

    m = Chain(x -> x ./ b)
    testjs(w, m, x)

    m = Chain(x -> x .- b)
    testjs(w, m, x)

    m = Chain(x -> exp.(x))
    testjs(w, m, x)

    m = Chain(x -> log.(x))
    testjs(w, m, x)

    m = Chain(x -> x .^ [2])
    testjs(w, m, x)

    m = Chain(x -> σ.(x))
    testjs(w, m, x)

    m = Chain(x -> tanh.(x))
    testjs(w, m, x)

    m = Chain(x -> relu.(x))
    testjs(w, m, x)

    m = Chain(x -> leakyrelu.(x))
    testjs(w, m, x)

    m = Chain(x -> copy.(x))
    testjs(w, m, x)
end

@testset "Conv" begin
    m = Chain(Conv((2, 2), 2=>2))
    x = rand(2,2,2,2)
    testjs(w, m, x)
end

@testset "maxpool" begin
    m = Chain(Conv((2, 2), 2=>2), x -> maxpool(x, (2, 2)))
    x = rand(4,4,2,2)
    testjs(w, m, x)
end

@testset "BatchNorm" begin
    m = Chain(BatchNorm(10))
    Flux.testmode!(m)
    x = rand(10, 1)
    testjs(w, m, x)
end

@testset "reshape" begin
    m = Chain(x -> reshape(x, :, size(x, 3)))
    x = rand(1, 2, 3)
    testjs(w, m, x)
end

@testset "LSTM" begin
    m = Chain(x -> Flux.gate(x, 4, 1))
    x = rand(10)
    testjs(w, m, x)

    m = Chain(LSTM(10, 10))
    x = rand(10)
    testjs(w, m, x)
end

@testset "mean" begin
    m = Chain(x -> mean(x, 1))
    x = rand(10, 10)
    testjs(w, m, x)
end

end
