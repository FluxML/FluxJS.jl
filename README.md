[![Build Status](https://travis-ci.org/FluxML/FluxJS.jl.svg?branch=master)](https://travis-ci.org/FluxML/FluxJS.jl)

# Flux.JS

Run [Flux](https://fluxml.github.io/) models in the browser, via
[tensorflow.js](https://js.tensorflow.org).

Note that if you get errors running this package, you may need to run `Pkg.checkout("ASTInterpreter2")`.

## JS Output

You can see what Flux.JS sees with `@code_js`, which works like `@code_typed` or
`@code_native`. Flux.JS simply accepts a function of arrays along with example
inputs, and generates JavaScript code for you. Here's the simplest possible
example:

```js
julia> x = rand(10)
10-element Array{Float64,1}:
 0.299338
 ⋮
 0.267917

julia> @code_js identity(x)
let model = (function () {
  let math = tf;
  function model(kinkajou) {
    return kinkajou;
  };
  model.weights = [];
  return model;
})();
flux.fetchWeights("model.bson").then((function (ws) {
  return model.weights = ws;
}));
```

You can see that there's some setup code as Flux.JS expects to load some weights
for a model. But the core of it is this function, which is exactly like the
`identity` function in Julia.

```js
function model(kinkajou) {
  return kinkajou;
};
```

Let's try something more interesting; `f` takes two arguments and multiplies
them.

```js
julia> f(W,x) = W*x

julia> @code_js f(rand(5,10),rand(10))
let model = (function () {
  let math = tf;
  function model(bear, giraffe) {
    return math.matrixTimesVector(bear, giraffe);
  };
  model.weights = [];
  return model;
})();
flux.fetchWeights("model.bson").then((function (ws) {
  return model.weights = ws;
}));
```

Because Flux models are just Julia functions, we can use the same macro with
them too. You'll now notice that the weights are being used.

```js
julia> m = Chain(Dense(10,5,relu),Dense(5,2),softmax)

julia> @code_js m(x)
let model = (function () {
  let math = tf;
  function badger(eland) {
    return math.add(math.matrixTimesVector(model.weights[0], eland), model.weights[1]);
  };
  function chimpanzee(mongoose) {
    return math.relu(math.add(math.matrixTimesVector(model.weights[2], mongoose), model.weights[3]));
  };
  function model(shark) {
    return math.softmax(badger(chimpanzee(shark)));
  };
  model.weights = [];
  return model;
})();
flux.fetchWeights("model.bson").then((function (ws) {
  return model.weights = ws;
}));
```

There is also early support for RNNs (we compile stateful models directly, no
unrolling).

```js
julia> m = Chain(RNN(10,5))

julia> @code_js m(x)
let model = (function () {
  let math = tf;
  let init = [0.017732, 0.00991122, -0.00712077, -0.00161244, -0.00232475];
  let states = init.slice();
  function nightingale(seal, mongoose) {
    return [seal, mongoose];
  };
  function cat(horse) {
    let weasel = math.tanh(math.add(math.add(math.matrixTimesVector(model.weights[0], horse), math.matrixTimesVector(model.weights[1], states[0])), model.weights[2]));
    let coati = nightingale(weasel, weasel);
    states[0] = coati[1];
    return coati[2];
  };
  function model(fish) {
    return cat(fish);
  };
  model.reset = (function () {
    states = init.slice();
    return;
  });
  model.weights = [];
  return model;
})();
flux.fetchWeights("model.bson").then((function (ws) {
  return model.weights = ws;
}));
```

In general, the more useful entry point to the package is `FluxJS.compile`.

```julia
julia> FluxJS.compile("mnist", m, rand(10))
```

This will produce two files in the current directory: (1) `mnist.js`, which
contains the same JavaScript code as above; (2) `mnist.bson`, which contains the
model weights in a JS-loadable format.

## Browser Setup

Firstly, you'll need the following scripts in your `<head>`. The `flux.js`
script can be found [here](lib/flux.js).

```html
<head>
  <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@0.9.0"></script>
  <script src="https://unpkg.com/bson/browser_build/bson.js"></script>
  <script src="flux.js"></script> <!-- Or embed the script directly -->
</head>
```

From here, you can either link the generated code as another script, or embed it
directly. In real applications you'll most likely want to wait on the
`fetchWeights` promise, to avoid trying to use the model before it's ready.

```html
<script>
let model = (function () {
  let math = tf;
  function model(kinkajou) {
    return kinkajou;
  };
  model.weights = [];
  return model;
})();
flux.fetchWeights("model.bson").then((function (ws) {
  return model.weights = ws;
}));
</script>
```

In the page, you can run the model from the dev tools.

```js
> x = tf.tensor([1,2,3,4,5,6,7,8,9,10])
  Tensor {isDisposed: false, size: 10, shape: Array(1), dtype: "float32", strides: Array(0), …}
> await model(x).data()
  Float32Array(25) [0.0262143611907959, -0.04852187633514404, …]
```

See the [tensorflow.js docs](https://js.tensorflow.org/api/latest/index.html) for
more information on how to work with its tensor objects.
