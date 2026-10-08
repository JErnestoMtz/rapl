# Examples

## Conway's Game of Life
- Conway's Game of Life is a well-known cellular automaton in which each generation of a population "evolves" from the previous one according to a set of predefined rules. This particular implementation is inspired in the very famous APL "one-liner" [implementation](https://aplwiki.com/wiki/Conway%27s_Game_of_Life).
```
cargo run --release --example conway_gol
```
<img src="https://github.com/JErnestoMtz/rapl/blob/main/graphics/gol.gif" width="300">

## Edge detection with FFT
- Small image processing example where we use the FFT and the [convolution theorem](https://en.wikipedia.org/wiki/Convolution_theorem) to extract information about the edges of an image.

```
cargo run --release --example image_fft --features fft
```

original image            |  Processed image
:-------------------------:|:-------------------------:
<img src="https://github.com/JErnestoMtz/rapl/blob/main/graphics/peppers.png" width="300"> |  <img src="https://github.com/JErnestoMtz/rapl/blob/main/graphics/pepper_edges.png" width="300">
## Convolution from primitives
- A conv2d layer and 2x2 max pooling built from the core algebra: `Win` in the slice grammar gives a zero-copy windows view, and the fused contraction folds each patch against the filters at O(output) memory. Runs two Sobel edge detectors over a synthetic image and prints the feature maps.
```
cargo run --release --example convolution
```

## Ising Model
- The [Ising model](https://en.wikipedia.org/wiki/Ising_model) is a mathematical model of interacting particles with binary (spin up or down) degrees of freedom, for example magnetic moments in a ferromagnetic material. This is a simple implementation of this model that produces an animation of the spin lattice in the terminal:
```
cargo run --release --example ising_model
```
<img src="https://github.com/JErnestoMtz/rapl/blob/main/graphics/Ising.gif" width="300">

## AlexNet-style training

[`alexnet.rs`](alexnet.rs) implements AlexNet's five-convolution, three-dense
network, including ReLU, overlapping max pooling, softmax cross-entropy, and
manual backpropagation. It uses NCHW images, `f32`, and an example-owned seeded
normal generator. No dependencies are needed. Layers propagate shape errors
with `?`; calling `backward` before `forward` is a programming error and panics.

```sh
# Compact network: all 19 layers, 67px inputs, 3,942 parameters, two classes.
# Prints every layer shape, then overfits four synthetic striped images.
cargo run --release --example alexnet --no-default-features

# Full widths: 227px inputs, 62,378,344 parameters, 1000 classes.
# Runs a forward pass; full-size training is opt-in because of its compute cost.
cargo run --release --example alexnet --no-default-features -- --full
cargo run --release --example alexnet --no-default-features -- --full --steps 12

# Finite-difference gradients, pooling overlaps/ties, and end-to-end learning.
cargo test --example alexnet --no-default-features
```

Convolution uses borrowed `Win` patches, axis permutation, and contraction,
without `im2col`. Fixed ranks check the layer mathematics; `Dyn` connects the
layers into a sequential network. The architecture is ungrouped, with no
dropout or local response normalization.

The compact demo reaches 100% training accuracy on four synthetic images; it
demonstrates backpropagation, not generalization. Override its default 80
optimization steps with `--steps N`.

## Multi-head self-attention

[`attention.rs`](attention.rs) is a pre-norm causal multi-head self-attention
block: layer norm, Q/K/V projections, head split, scaled scores, causal mask,
softmax, head merge, output projection and residual. Inputs are deterministic,
so the printed checksum is reproducible.

```sh
cargo run --example attention
cargo test --example attention
```
