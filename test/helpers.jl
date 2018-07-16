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
        $result[] = model(x).transpose().dataSync()
    end)
end

function testjs(model, x)
    w = Blink.Window(Dict(:show => false))
    Blink.body!(w, dom"div"())
    sleep(5)

    libs = [
        "//unpkg.com/bson@2.0.8/browser_build/bson.js",
        "//cdnjs.cloudflare.com/ajax/libs/tensorflow/0.11.7/tf.js"]
    fluxjs = [normpath("$(@__DIR__)/../lib/flux.js")]

    loadseq(w, Dict("files"=>libs, "names"=>["BSON", "tf"]), Dict("files"=>fluxjs, "names"=>["fluxjs"]))

    s = Scope()
    r = Observable(s, "result", Dict())
    output = Channel{Dict}(1)

    mjs = modeljs(model, x, r)
    onimport(s, mjs)

    p = x -> round(x, 5)

    on(r) do o
        put!(output, o)
    end
    Blink.body!(w, s)

    sleep(5)
    res = Flux.data(model(x))
    o = take!(output)
    resjs = Array{eltype(res),1}(length(o))
    for i in keys(o)
        resjs[parse(i) + 1] = o[i]
    end

    @test p.(res) == p.(resjs)
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
    sleep(5)
    loadseq(w, args...)
end
