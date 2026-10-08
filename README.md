# rapl
[![Documentation](https://docs.rs/rapl/badge.svg)](https://docs.rs/rapl)
[![Crate](https://img.shields.io/crates/v/rapl.svg)](https://crates.io/crates/rapl)

**Enjoyable, composable, hackable N-dimensional arrays for Rust.**

`rapl` combines the familiar parts of NumPy with a lot of inspiration from array
programming languages like APL and BQN. Ranks are part of the type, so rank
mistakes are compile errors.

`rapl` has a small core, and every operation can be built from a small set of
composable primitives.

```rust
use rapl::*;

fn main() -> Result<(), DimError> {
    let x = Ndarr::from([3, 1, 4, 1, 5, 9, 2, 6]);

    let w = x.slice(s![Win(3)])?;          // every window of 3: a [6, 3] view, no copy
    println!("{w}");
    // ┌→────┐
    // ↓3 1 4│
    // │1 4 1│
    // │4 1 5│
    // │1 5 9│
    // │5 9 2│
    // │9 2 6│
    // └~────┘

    let peaks = w.reduce(1, i32::max)?;    // 4 4 5 9 9 9

    let hops = w.slice(s![..;2, ..])?;     // a stride is just a step on the positions
    // ┌→────┐
    // ↓3 1 4│
    // │4 1 5│
    // │5 9 2│
    // └~────┘

    // The same in 2-D: every 2×2 patch of an image, then max pooling.
    let img = Ndarr::from([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]);
    let pooled = img
        .slice(s![Win(2), Win(2)])?        // [2, 2, 3, 2]: row, kh, col, kw
        .reduce(3, i32::max)?
        .reduce(1, i32::max)?;
    // ┌→───────┐
    // ↓ 6  7  8│
    // │10 11 12│
    // └~───────┘
    Ok(())
}
```

> `rapl` is in early development. The API is still settling, and speed is not yet
> a priority (see [Philosophy](#philosophy)), so it is not yet recommended for
> speed-sensitive applications.

## What makes rapl different

**Rank generic.** Every operation works on arrays of any rank: a vector, a
matrix, a batch of images, a stack of attention heads. The rank lives in the
type (`Ndarr<f32, U3>`), so a rank mismatch is a compile error.

**Type generic where possible.** Elements are any `T`, not just floats:
strings, chars, tuples, your own structs. Arithmetic needs arithmetic and `sin`
needs a float. Everything else (slicing, permuting, mapping, folding, outer
products) works for any element type.

**Enjoyable, composable, hackable.** `rapl` is not the fastest ndarray library
in Rust and does not try to be. It tries to be pleasant to write, read and
modify: a small core, **three small dependencies**, and conveniences defined as short
compositions of public primitives rather than private kernels.

## A taste

A 2-D convolution from windows and one fused contraction:

```rust
// image: [h, w, c_in], filters: [kh, kw, c_in, c_out]
let patches = image
    .slice(s![Win(3), Win(3), ..])?       // [h', 3, w', 3, c_in], a view
    .permute_axes(&[0, 2, 1, 3, 4])?;     // [h', w', 3, 3, c_in]
let features = patches.contract(&filters, U3::new(), |x, w| x * w, |a, b| a + b)?;
```

The [examples](https://github.com/JErnestoMtz/rapl/tree/main/examples) include
AlexNet with hand-written backpropagation, multi-head attention, an
ultra-compact APL-style Conway's Game of Life, an Ising model, and FFT edge
detection.

## A tour

Snippets assume `use rapl::*;` and a function that returns `Result`, so `?`
works.

### Creating arrays

```rust
let words = Ndarr::from(vec!["a", "b", "c"]);            // any element type
let matrix = Ndarr::from([[1, 2], [3, 4]]);              // rank from the literal
let range = Ndarr::from(1..7).reshape([2, 3])?;
let chars = Ndarr::from("Hello rapl!");                  // Ndarr<char, U1>

let zeros = Ndarr::<f64, U3>::zeros([2, 3, 4]);
let filled = Ndarr::fill("a", [5]);
let line = Ndarr::linspace(0.0, 1.0, 5);

// `from_fn` passes each element's coordinates:
let identity = Ndarr::from_fn([3, 3], |ix| i32::from(ix[0] == ix[1]));
// ┌→────┐
// ↓1 0 0│
// │0 1 0│
// │0 0 1│
// └~────┘
```

Random arrays use the same `from_fn` with a generator you own; `rapl` does not
choose a PRNG.

### Element-wise math and broadcasting

```rust
// Shapes broadcast from the right, as in NumPy, and complex numbers are native:
let a = Ndarr::from([1 + 1.i(), 2 + 1.i()]);
let b = Ndarr::from([[1, 2], [3, 4]]);
assert_eq!(a + b - 2, Ndarr::from([[1.i(), 2 + 1.i()], [2 + 1.i(), 4 + 1.i()]]));

let x = Ndarr::linspace(-1.0_f64, 1.0, 5);
let y = x.sin() * 2.0 + x.tanh();
let relu = x.map(|v| v.max(0.0));                 // anything else is one `map` away

// Combine two arrays, of different element types if needed:
let grid = Ndarr::from([[1, 2, 3], [4, 5, 6]]);
let labels = grid.zip_with(&Ndarr::from(vec!["a", "b", "c"]), |n, s| format!("{s}{n}"))?;
// ┌→───────┐
// ↓a1 b2 c3│
// │a4 b5 c6│
// └~───────┘
```

### Indexing, slicing and views

```rust
let mut a = Ndarr::from([[0, 1, 2, 3], [4, 5, 6, 7]]);
assert_eq!(a[[1, 2]], 6);                          // fixed or dynamic rank alike

let row = a.slice(s![1, ..])?;                     // an integer drops the axis
let evens = a.slice(s![.., ..;2])?;                // ranges with steps
let reversed = a.slice(s![.., ..;-1])?;            // negative steps, as in NumPy
let start = 1usize;
let tail = a.slice(s![0, start..])?;               // any integer type, no casts

a.slice_mut(s![.., 0])?.map_in_place(|v| v * 10);  // mutable views, same syntax
```

Views are O(1): `slice`, `t_view`, windows and broadcasts share the original
buffer. `permute_axes` and `reshape` consume their input and keep its storage,
so they never copy; `to_owned_array()` is the explicit copy.

### Reductions, scans and cells

```rust
use std::ops::Add;

let m = Ndarr::from([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
let col_sums = m.reduce(0, f64::add)?;                  // [5, 7, 9], axis removed
let row_max = m.reduce(Keep(1), f64::max)?;             // [[3], [6]], axis kept
let centered = &m - &row_max;                           // so it broadcasts back
let running = m.scan_axis(1, ScanDirection::Forward, |acc, x| acc + x)?;

// Cells: any function over the trailing axes, e.g. the max of each window.
let windows = m.slice(s![Win(2), Win(2)])?.permute_axes(&[0, 2, 1, 3])?;
let window_max = windows.map_cells(U2::new(), |w| {
    w.iter_elems().copied().fold(f64::NEG_INFINITY, f64::max)
})?;
```

### Products

```rust
let a = Ndarr::from(1..7).reshape([2, 3])?;
let b = Ndarr::from(1..7).reshape([3, 2])?;
let product = a.mat_mul(&b)?;

// `mat_mul` batches leading axes like NumPy's `@`:
let batch = Ndarr::from(1..13).reshape([2, 2, 3])?;
assert_eq!(batch.mat_mul(&b)?.shape(), &[2, 2, 2]);

// The general form: any number of contracted axes, any combining functions.
let tensordot = a.contract(&b, U1::new(), |x, y| x * y, |x, y| x + y)?;

// Outer products combine every pair, for any element type:
let suits = Ndarr::from(vec!["♣", "♠", "♥", "♦"]);
let ranks = Ndarr::from(vec!["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]);
let deck = ranks.outer_product(&suits, |rank, suit| format!("{rank}{suit}"))?;
assert_eq!(deck.len(), 52);
println!("{deck}");
// ┌→──────────────┐
// ↓2♣  2♠  2♥  2♦ │
// │3♣  3♠  3♥  3♦ │
// │4♣  4♠  4♥  4♦ │
// │5♣  5♠  5♥  5♦ │
// │6♣  6♠  6♥  6♦ │
// │7♣  7♠  7♥  7♦ │
// │8♣  8♠  8♥  8♦ │
// │9♣  9♠  9♥  9♦ │
// │10♣ 10♠ 10♥ 10♦│
// │J♣  J♠  J♥  J♦ │
// │Q♣  Q♠  Q♥  Q♦ │
// │K♣  K♠  K♥  K♦ │
// │A♣  A♠  A♥  A♦ │
// └~──────────────┘
```

### Rank checking, static and dynamic

```rust
let m = Ndarr::from([[1, 2], [3, 4]]);
let v: Ndarr<i32, U1> = m.reduce(0, |a, b| a + b)?;   // U2 -> U1, checked at compile time
let [rows, cols] = m.dim().to_array();               // destructure a fixed-rank shape

let d = m.clone().into_dyn();                         // rank known only at runtime
let w: Ndarr<i32, Dyn> = d.reduce(0, |a, b| a + b)?;  // same API, checked at runtime
```

### Complex numbers and FFT

```rust
let z = 1 + 2.i();
assert_eq!(z - 3, -2 + 2.i());

let shifted = Ndarr::from([1, 2, 3]) + -1 + 2.i();    // complex arrays
assert_eq!(shifted.im(), Ndarr::from([2, 2, 2]));

// 1-D and 2-D FFT with the `fft` feature, implemented in rapl itself:
let signal = Ndarr::linspace(-10.0, 10.0, 100).sin();
let spectrum = signal.to_complex().fft();
```

## Philosophy

- **Small core, everything composes.** `rapl` exposes a handful of structural
  and higher-order primitives: cells, windows, scans, reduce, contraction.
  Conveniences are short definitions over them, so improving a primitive
  improves everything built on it.
- **Clarity over speed.** There are no SIMD kernels, BLAS bindings or
  special-cased fast paths. If a clear composition of primitives can express
  something, that composition is the implementation. For heavy production
  workloads, use a performance-oriented crate; `rapl` is meant for exploring,
  prototyping, teaching and scripting.
- **Minimal dependencies.** Every dependency has to earn its place. Today
  there are three small, widely used crates and nothing else at runtime:
  `typenum` and `generic-array` carry the compile-time rank system, and
  `num-traits` provides the numeric traits. Features like the FFT are
  implemented in `rapl` itself rather than pulled in.
- **Hackable.** No `unsafe`, and a codebase meant to be read. A missing
  function is usually a `map`, a `reduce` or a `contract` away, and we would
  rather document the composition than grow the catalog.

## Getting started

```sh
cargo add rapl
# or, with the FFT:
cargo add rapl --features fft
```

Complex numbers are enabled by default. Contributions and issues are welcome.
