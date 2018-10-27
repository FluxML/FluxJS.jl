function modeljs(model, x, result::WebIO.Observable)
    code, weights = FluxJS.compile(model, x)
    code = join(split(code, "\n")[1:end-4],"\n") # remove fetch statement
    code = WebIO.JSString(code)

    JSExpr.@js (() -> begin
        $(code)
        JSExpr.@var weights = $weights.map(e -> begin
            tf.tensor(e)
        end)
        JSExpr.@var x = tf.tensor($x)
        model.weights = weights
        $result[] = model(x).dataSync()
    end)
end

function Base.take!(o::Channel,timeout::Int)
    @async begin
        sleep(timeout)
        return nothing
    end
    return take!(o)
end

function compare(res, out)
    out == nothing && return false
    resjs = Array{eltype(res),1}(undef, length(out))
    for i in keys(out)
        resjs[Main.parse(i) + 1] = out[i]
    end
    return all(x -> x, abs.(res .- resjs) .< 10.0^(-5))
end

function testjs(w, model, x)
    s = Scope()
    r = Observable(s, "result", Dict())
    output = Channel{Dict}(1)
    mjs = modeljs(model, x, r)
    onimport(s, mjs)
    on(r) do o
        put!(output, o)
    end
    Blink.body!(w, s)
    res = [Flux.data(model(x))...]
    @test compare(res, take!(output, 1))
end

loadseq(w) = nothing

function loadseq(w, config)
    s = Scope(imports=config["files"])
    onimport(s, JSExpr.@js ()->begin
        JSExpr.@var names = $(config["names"])
        JSExpr.@var modules = arguments
        names.forEach((e, i) -> begin
            window[e] = modules[i]
        end)
    end)
    Blink.body!(w, s)
end

function loadseq(w, config, args...)
    loadseq(w, config)
    loadseq(w, args...)
end

function setupWindow()
    w = Blink.Window(Dict(:show => false))
    libs = [
        "//unpkg.com/bson@2.0.8/browser_build/bson.js",
        "//cdnjs.cloudflare.com/ajax/libs/tensorflow/0.11.7/tf.js"]
    fluxjs = [normpath("$(@__DIR__)/../lib/flux.js")]
    loadseq(w, Dict("files"=>libs, "names"=>["BSON", "tf"]), Dict("files"=>fluxjs, "names"=>["fluxjs"]))
    return w
end
