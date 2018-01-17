# Flux.JS

WIP model export from [Flux](https://fluxml.github.io/) to [deeplearn.js](https://deeplearnjs.org/). This should also serve as a proof-of-concept for exporting to other backends, like TensorFlow Lite or ONNX.

A model like this:

```julia
d = Dense(5, 5)
model = Chain(d, x->σ.(x), d, softmax)
```

Produces output like this:

```js
math = dl.ENV.math;
c997 = dl.Array2D.new([5, 5], [0.24327, 0.413138, -0.0474187, -0.0720937, -1.60018, 0.321784, 1.52317, -1.03032, -0.956021, -1.28895, -0.251022, 1.24529, 0.919683, 1.62069, 1.7547, 0.00799773, -1.28945, -0.716793, -1.19455, 0.245558, 0.409453, -0.572654, 0.179335, -1.25914, 0.483916]);
c998 = dl.Array1D.new([0.304343, 0.100034, -1.45973, -0.775166, 0.526972]);
function c1000(x999) {
  return math.add(math.matrixTimesVector(c997, x999), c998);
};
function model(x1001) {
  return math.softmax(c1000(math.sigmoid(c1000(x1001))));
};
```
