function inline_blocks(ex)
  ex = MacroTools.postwalk(ex) do ex
    @capture(ex, x_ = (body__; y_)) || return ex
    unblock(:($(body...); $x = $y))
  end
  ex = MacroTools.postwalk(ex) do ex
    @capture(shortdef(ex), name_ = (args__,) -> body_) || return ex
    :(function $name($(args...)) $body end)
  end
end

function valid_names(ex)
  MacroTools.prewalk(ex) do x
    x isa Symbol ? Symbol(replace(String(x), "#", "")) : x
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

preamble = quote
  math = dl.ENV.math
end

prepare(ex) = quote
  $preamble
  model = $(ex)
end |> insert_returns |> valid_names |> inline_blocks |> MacroTools.flatten |> MacroTools.striplines

compile(v::IVertex) = v |> lower |> DataFlow.syntax |> prepare |> jsexpr

# JS output

function jsexpr_joined(io::IO, xs, delim=", ")
  isempty(xs) && return
  for i = 1:length(xs)-1
    jsexpr(io, xs[i])
    print(io, delim)
  end
  jsexpr(io, xs[end])
end

jsexpr_joined(xs, delim=", ") = sprint(jsexpr_joined, xs, delim)

function call_expr(io, f, args...)
  f in [:(==), :+, :-, :*, :/, :%] &&
    return print(io, "(", jsexpr_joined(args, string(f)), ")")
  jsexpr(io, f)
  print(io, "(")
  jsexpr_joined(io, args)
  print(io, ")")
end

function func_expr(io, args, body, level = 0)
  named = isexpr(args, :call)
  named || print(io, "(")
  print(io, "function ")
  if named
    print(io, args.args[1])
    args = Expr(:tuple, args.args[2:end]...)
  end
  print(io, "(")
  isexpr(args, Symbol) ? print(io, args) : join(io, args.args, ",")
  println(io, ") {")
  jsexpr(io, block(body), level = level+1)
  print(io, "}")
  named || print(io, ")")
end

function if_expr(io, xs; level = 0)
    if length(xs) >= 2  # we have an if
        print(io, "if (")
        jsexpr(io, xs[1])
        println(io, ") {")
        jsexpr(io, block(xs[2]), level = level+1)
        print(io, "  "^level, "}")
    end

    if length(xs) == 3  # Also have an else
        println(io, " else {")
        jsexpr(io, block(xs[3]), level = level+1)
        print(io, "  "^level, "}")
    end
end

function jsexpr(io::IO, x; level = 0)
  if isexpr(x, :block)
    for x in x.args
      print(io, "  "^level)
      jsexpr(io, x, level = level)
      println(io, ";")
    end
  elseif isexpr(x, :(=))
    jsexpr(io, x.args[1])
    print(io, " = ")
    jsexpr(io, x.args[2])
  elseif isexpr(x, :call)
    call_expr(io, x.args...)
  elseif isexpr(x, :->) || isexpr(x, :function)
    func_expr(io, x.args...)
  elseif isexpr(x, :if)
    if_expr(io, x.args, level = level)
  elseif isexpr(x, :return)
    print(io, "return ")
    jsexpr(io, x.args[1], level = level)
  else
    print(io, x)
  end
end

jsexpr(io::IO, x::Tuple; level = 0) = jsexpr(io, [x...], level = level)

jsexpr(x) = sprint(jsexpr, x)
