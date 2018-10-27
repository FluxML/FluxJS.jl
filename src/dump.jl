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

function func_expr(io, args, body; level = 0)
  named = isexpr(args, :call)
  named || print(io, "(")
  print(io, "function ")
  if named
    print(io, args.args[1])
    args = Expr(:tuple, args.args[2:end]...)
  end
  print(io, "(")
  isexpr(args, Symbol) ? print(io, args) : join(io, args.args, ", ")
  println(io, ") {")
  jsexpr(io, block(body), level = level+1)
  print(io, "  "^level, "}")
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
    for xe in x.args
      print(io, "  "^level)
      jsexpr(io, xe, level = level)
      println(io, ";")
    end
  elseif isexpr(x, :(=))
    isexpr(x.args[1], Symbol) && print(io, "let ")
    jsexpr(io, x.args[1])
    print(io, " = ")
    jsexpr(io, x.args[2], level = level)
  elseif isexpr(x, :.) && x.args[2] isa QuoteNode
    print(io, x.args[1], ".", x.args[2].value)
  elseif isexpr(x, :call)
    call_expr(io, x.args...)
  elseif isexpr(x, :tuple)
    print(io, "[")
    jsexpr_joined(io, x.args, ", ")
    print(io, "]")
  elseif @capture(x, global foo_ = bar_)
    jsexpr(io, foo)
    print(io, " = ")
    jsexpr(io, bar)
  elseif isexpr(x, :->) || isexpr(x, :function)
    func_expr(io, x.args..., level = level)
  elseif isexpr(x, :if)
    if_expr(io, x.args, level = level)
  elseif isexpr(x, :return)
    x.args[1] == nothing && return print(io, "return")
    print(io, "return ")
    jsexpr(io, x.args[1], level = level)
  elseif isexpr(x, :ref)
    jsexpr(io, x.args[1])
    print(io, "[")
    jsexpr(io, x.args[2])
    print(io, "]")
  else
    print(io, x)
  end
end

jsexpr(io::IO, x::Tuple; level = 0) = jsexpr(io, [x...], level = level)
jsexpr(io::IO, x::AbstractString; level = 0) = show(io, x)

jsexpr(x) = sprint(jsexpr, x)
