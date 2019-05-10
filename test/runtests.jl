using Flux, FluxJS, DataFlow, MacroTools
using FluxJS: traceλ
using JSExpr, JSON, WebIO
using Blink
using Test
using Flux.Tracker: data

atomshell = Blink.AtomShell.isinstalled()

if !atomshell
    Blink.AtomShell.install()
end

include("./helpers.jl")

@testset "FluxJS" begin

m = Dense(10,5)
v = traceλ(m, rand(10))
ex = prettify(DataFlow.syntax(v))

@test @capture ex _ -> (+).(matVecMul(_,_),_)

w = setupWindow()

x = rand(Float32, 2)
@testset "identity" begin
    testjs(w, identity, x)
end

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
    m = Chain(Dense(2, 2))
    testjs(w, m, x)
end

@testset "Chain" begin
    m = Chain(identity, identity)
    testjs(w, m, x)
end

@testset "broadcast" begin
    b = ones(2)

    m = x -> x .+ b
    testjs(w, m, x)

    m = x -> x .* b
    testjs(w, m, x)

    m = x -> x ./ b
    testjs(w, m, x)

    m = x -> x .- b
    testjs(w, m, x)

    m = x -> exp.(x)
    testjs(w, m, x)

    m = x -> log.(x)
    testjs(w, m, x)

    m = x -> x .^ [2]
    testjs(w, m, x)

    m = x -> σ.(x)
    testjs(w, m, x)

    m = x -> tanh.(x)
    testjs(w, m, x)

    m = x -> relu.(x)
    testjs(w, m, x)

    m = x -> leakyrelu.(x)
    testjs(w, m, x)

    m = x -> copy.(x)
    testjs(w, m, x)
end

@testset "Conv" begin
    m = Chain(Conv((2, 2), 2=>2))
    x = rand(Float32, 2,2,2,2)
    testjs(w, m, x)
end

@testset "maxpool" begin
    m = Chain(Conv((2, 2), 2=>2), MaxPool((2, 2)))
    x = rand(Float32, 4,4,2,2)
    testjs(w, m, x)
end

@testset "BatchNorm" begin
    m = Chain(BatchNorm(10))
    Flux.testmode!(m)
    x = rand(10, 1)
    testjs(w, m, x)
end

@testset "reshape" begin
    m = x -> reshape(x, :, size(x, 3))
    x = rand(1, 2, 3)
    testjs(w, m, x)
end

@testset "RNN" begin
    m = Chain(RNN(10, 10))
    x = rand(10)
    testjs(w, m, x)
end

@testset "LSTM" begin
    m = x -> Flux.gate(x, 4, 1)
    x = rand(10)
    testjs(w, m, x)

    m = Chain(LSTM(10, 10))
    x = rand(10)
    testjs(w, m, x)
end

end

if !atomshell
    Blink.AtomShell.uninstall()
end
