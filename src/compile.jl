struct Meth
  func
  args::Tuple
  graph
end

jscall(args...) = vcall(args...)

jscall(::typeof(identity), x) = x

cvalue(v) = DataFlow.isconstant(v) ? v.value.value : v

function inline_methods(v)
  DataFlow.prewalkλ(v) do v
    cvalue(v) isa Meth ? vcall(identity, cvalue(v).graph) : v
  end
end

function lower(v)
  v = inline_methods(v)
  v = DataFlow.prewalk(v -> DataFlow.islambda(v) ? DataFlow.λopen(v) : v, v)
  v = DataFlow.postwalk(v) do v
    v.value isa DataFlow.Line && return v[1]
    DataFlow.iscall(v) || return v
    jscall(cvalue.(v[:])...)
  end
  v = DataFlow.postwalk(v -> v.value isa DataFlow.OLambda ? DataFlow.λclose(v) : v, v)
  DataFlow.prewalkλ(v) do v
    DataFlow.iscall(v, control) ? v[2] : v
  end
end

function liftweights(v, weights=[])
  v = DataFlow.prewalkλ(v) do x
    cvalue(x) isa AbstractArray || return x
    push!(weights, Float32.(data(cvalue(x))))
    DataFlow.constant(:(model.weights[$(length(weights)-1)]))
  end
  v, weights
end

liftweights(c::Nothing, w=[]) = (c, w)

# Julia Code Stuff

using MacroTools: alias_gensyms, flatten, striplines

function inline_blocks(ex)
  ex = MacroTools.postwalk(ex) do ex
    @capture(ex, x_ = (body__; y_)) || return ex
    unblock(:($(body...); $x = $y))
  end
  ex = MacroTools.postwalk(ex) do ex
    @capture(ex, name_Symbol = (args__,) -> body_) || return ex
    :(function $name($(args...)) $body end)
  end
end

function insert_return(ex)
  isexpr(ex, :block) ? :($(ex.args[1:end-1]...);$(insert_return(ex.args[end]))) :
  isexpr(ex, :if) ? Expr(:if, ex.args[1], map(x -> insert_return(x), ex.args[2:end])...) :
  isexpr(ex, :return) ? ex :
  :(return $ex)
end

function insert_returns(ex)
  MacroTools.prewalk(ex) do x
    isexpr(x, :->, :function) ?
      Expr(x.head, x.args[1], insert_return(x.args[2])) :
      x
  end
end

function prepare(ex, name, states = nothing)
  decl_state = states == nothing ? :(;) :
    :(init = []; states = [])
  state_setup = states == nothing ? :(;) :
    :(global init = $states; global states = init.slice())
  reset_method = states == nothing ? :(;) :
    :(model.reset = () -> (global states = init.slice(); return);)
  get_states = states == nothing ? :(;) : :(model.getStates = () -> (return states);)
  set_weights = quote
    model.setWeights = (ws) -> begin
      model.weights = ws;
      $(state_setup.args...)
      return;
    end
  end
  quote
    model = (() -> begin
      math = tf;
      model.weights = []
      $(decl_state.args...)
      model = $(ex)
      $(reset_method.args...)
      $(get_states.args...)
      $(set_weights.args...)
      model
    end)()
    flux.fetchWeights($"$name.bson").then(model.setWeights)
  end |> insert_returns |> inline_blocks |> alias_gensyms |> flatten |> striplines
end

function compile(v::IVertex, name, states = [])
  statesv = states == [] ? nothing :
    unwrap((states...,)) |> lower
  v, weights = liftweights(lower(v))
  statesv, weights = liftweights(statesv, weights)
  statesv == nothing || (statesv = DataFlow.syntax(statesv))
  jsexpr(prepare(DataFlow.syntax(v), name, statesv)), weights
end

function compile(f, args...; name = "model")
  ctx = Trace()
  v = traceλ(f, args..., meta = ctx)
  compile(v, name, ctx.states)
end

function compile(name::AbstractString, f, args...)
  code, weights = compile(f, args..., name = name)
  open(io -> write(io, code), "$name.js", "w")
  BSON.@save "$name.bson" weights
  return
end

macro code_js(ex)
  @capture(ex, f_(args__)) || error("@code_js f(args...)")
  quote
    Text(compile($(esc(f)), $(esc.(args)...))[1])
  end
end
