//! AlexNet-style forward/backward passes composed from rapl primitives.
//! NCHW images; convolution weights are [C, kh, kw, F] for trailing-axis contraction.
//! Run the compact learning demo with `cargo run --release --example alexnet`.
//! `--full` selects the supplied 227px / 1000-class architecture (forward only).
//! Add `--steps N` to train either size. No datasets or extra dependencies needed.

use rapl::*;
use std::ops::Add;

type Tensor = Ndarr<f32, Dyn>;
type Image = Ndarr<f32, U4>;

// Example-owned generator: Box–Muller normal draws for He initialization.
struct Rng(u64);
impl Rng {
    fn uniform(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 40) as f32 + 0.5) / 16_777_216.0
    }
    fn normal(&mut self) -> f32 {
        (-2.0 * self.uniform().ln()).sqrt() * (std::f32::consts::TAU * self.uniform()).cos()
    }
}

/// Zero padding is a filled array plus assignment into its interior.
fn pad(x: Image, amount: usize) -> Result<Image, DimError> {
    if amount == 0 {
        return Ok(x);
    }
    let [n, c, h, w] = x.dim().to_array();
    let mut padded = Ndarr::zeros([n, c, h + 2 * amount, w + 2 * amount]);
    padded
        .slice_mut(s![.., .., amount..amount + h, amount..amount + w])?
        .zip_with_in_place(&x, |_, value| *value)?;
    Ok(padded)
}

/// [N, C, H, W] -> [N, oh, ow, C, kh, kw], all borrowed. No im2col allocation.
fn patches(x: &Image, k: usize, stride: usize) -> Result<NdView<'_, f32, U6>, DimError> {
    x.slice(s![.., .., Win(k), Win(k)])? // [N, C, oh, kh, ow, kw]
        .permute_axes(&[0, 2, 4, 1, 3, 5])? // -> [N, oh, ow, C, kh, kw]
        .slice(s![.., ..;stride, ..;stride, .., .., ..])
}

/// Shared overlap accumulation for convolution and pooling backpropagation:
/// the input positions one kernel offset touched, every `stride` from it.
/// Different offsets may overlap.
fn add_patch<B: Buffer<f32>>(
    dx: &mut Image,
    values: &Ndarr<f32, U4, B>,
    row: usize,
    col: usize,
    stride: usize,
) -> Result<(), DimError> {
    let [_, _, height, width] = values.dim().to_array();
    let rows = row..row + height * stride;
    let cols = col..col + width * stride;
    dx.slice_mut(s![.., .., rows;stride, cols;stride])?
        .zip_with_in_place(values, |a, b| a + b)?;
    Ok(())
}

/// Row-major position and value of the first maximum. Strict `>` keeps the
/// first winner at ties, as in NumPy's argmax.
fn argmax<'a>(values: impl Iterator<Item = &'a f32>) -> (usize, f32) {
    values
        .copied()
        .enumerate()
        .reduce(|best, item| if item.1 > best.1 { item } else { best })
        .expect("argmax of an empty cell")
}

// Only the sequential composition erases rank. Each spatial/dense layer restores
// its fixed rank at the boundary, so its tensor expressions are checked statically.
// Shape errors propagate; backward before forward is a programming error.
trait Layer {
    fn forward(&mut self, x: Tensor) -> Result<Tensor, DimError>;
    fn backward(&mut self, gradient: Tensor, lr: f32) -> Result<Tensor, DimError>;
    fn n_params(&self) -> usize {
        0
    }
}

struct Conv2D {
    weights: Image,
    bias: Ndarr<f32, U1>,
    stride: usize,
    padding: usize,
    input: Option<Image>,
}
impl Conv2D {
    fn new(
        c: usize,
        filters: usize,
        k: usize,
        stride: usize,
        padding: usize,
        rng: &mut Rng,
    ) -> Self {
        let scale = (2.0 / (c * k * k) as f32).sqrt();
        Self {
            weights: Ndarr::from_fn([c, k, k, filters], |_| rng.normal() * scale),
            bias: Ndarr::zeros([filters]),
            stride,
            padding,
            input: None,
        }
    }
}
impl Layer for Conv2D {
    fn forward(&mut self, x: Tensor) -> Result<Tensor, DimError> {
        let x = pad(x.into_ranked::<U4>()?, self.padding)?;
        // [N, oh, ow, C, kh, kw] with [C, kh, kw, F] -> [N, oh, ow, F]
        let windows = patches(&x, self.weights.shape()[1], self.stride)?;
        let y = windows.contract(&self.weights, U3::new(), |x, w| x * w, |a, b| a + b)?;
        // -> [N, F, oh, ow]: the owned sum is permuted in place, not copied.
        let out = (&y + &self.bias).permute_axes(&[0, 3, 1, 2])?;
        self.input = Some(x);
        Ok(out.into_dyn())
    }

    fn backward(&mut self, gradient: Tensor, lr: f32) -> Result<Tensor, DimError> {
        let x = self.input.take().expect("forward before backward");
        let gradient = gradient.into_ranked::<U4>()?; // [N, F, oh, ow]
        let d = gradient.view().permute_axes(&[0, 2, 3, 1])?; // -> [N, oh, ow, F]
        let k = self.weights.shape()[1];
        // [N, oh, ow, C, kh, kw] -> [C, kh, kw, N, oh, ow], with d -> [C, kh, kw, F]
        let dw = patches(&x, k, self.stride)?
            .permute_axes(&[3, 4, 5, 0, 1, 2])?
            .contract(&d, U3::new(), |x, d| x * d, |a, b| a + b)?;
        // Sum over N, oh, ow; the highest axis goes first so the rest keep their index.
        let db = gradient
            .reduce(3, f32::add)?
            .reduce(2, f32::add)?
            .reduce(0, f32::add)?;

        // The adjoint of patch extraction: accumulate into strided slices.
        // Contract one kernel offset at a time, avoiding a full dcols tensor.
        let mut dx = Ndarr::zeros(x.dim().clone());
        for i in 0..k {
            for j in 0..k {
                // d [N, oh, ow, F] with w^T [F, C] -> [N, oh, ow, C] -> [N, C, oh, ow]
                let w = self.weights.slice(s![.., i, j, ..])?; // [C, F]
                let contribution =
                    d.contract(&w.t_view(), U1::new(), |d, w| d * w, |a, b| a + b)?;
                let contribution = contribution.permute_axes(&[0, 3, 1, 2])?;
                add_patch(&mut dx, &contribution, i, j, self.stride)?;
            }
        }
        // The loss already averages over the batch. Do not divide by N again.
        self.weights.zip_with_in_place(&dw, |w, g| w - lr * g)?;
        self.bias.zip_with_in_place(&db, |b, g| b - lr * g)?;
        let [_, _, h, w] = dx.dim().to_array();
        let p = self.padding;
        let unpadded = dx.slice(s![.., .., p..h - p, p..w - p])?;
        Ok(unpadded.to_owned_array().into_dyn())
    }
    fn n_params(&self) -> usize {
        self.weights.len() + self.bias.len()
    }
}

#[derive(Default)]
struct ReLU(Option<Ndarr<bool, Dyn>>);
impl Layer for ReLU {
    fn forward(&mut self, mut x: Tensor) -> Result<Tensor, DimError> {
        self.0 = Some(x.map(|v| *v > 0.0));
        x.map_in_place(|v| v.max(0.0));
        Ok(x)
    }
    fn backward(&mut self, d: Tensor, _: f32) -> Result<Tensor, DimError> {
        let keep = self.0.take().expect("forward before backward");
        d.zip_with(&keep, |d, keep| if *keep { *d } else { 0.0 })
    }
}

struct MaxPool2D {
    k: usize,
    stride: usize,
    cache: Option<(Dim<U4>, Ndarr<usize, U4>)>,
}
impl MaxPool2D {
    fn new(k: usize, stride: usize) -> Self {
        Self {
            k,
            stride,
            cache: None,
        }
    }
}
impl Layer for MaxPool2D {
    fn forward(&mut self, x: Tensor) -> Result<Tensor, DimError> {
        let x = x.into_ranked::<U4>()?;
        // Each [kh, kw] window cell yields its value and row-major winner index.
        let winners = patches(&x, self.k, self.stride)?
            .permute_axes(&[0, 3, 1, 2, 4, 5])? // -> [N, C, oh, ow, kh, kw]
            .map_cells(U2::new(), |window| argmax(window.iter_elems()))?;
        self.cache = Some((x.dim().clone(), winners.map(|&(at, _)| at)));
        Ok(winners.map(|&(_, value)| value).into_dyn())
    }
    fn backward(&mut self, gradient: Tensor, _: f32) -> Result<Tensor, DimError> {
        let (shape, arg) = self.cache.take().expect("forward before backward");
        let d = gradient.into_ranked::<U4>()?;
        let mut dx = Ndarr::zeros(shape);
        for i in 0..self.k {
            for j in 0..self.k {
                let contribution =
                    d.zip_with(&arg, |d, &at| if at == i * self.k + j { *d } else { 0.0 })?;
                add_patch(&mut dx, &contribution, i, j, self.stride)?;
            }
        }
        Ok(dx.into_dyn())
    }
}

#[derive(Default)]
struct Flatten(Vec<usize>);
impl Layer for Flatten {
    fn forward(&mut self, x: Tensor) -> Result<Tensor, DimError> {
        self.0 = x.shape().to_vec();
        let rows = x.shape()[0];
        let columns = x.len() / rows;
        Ok(x.reshape([rows, columns])?.into_dyn())
    }
    fn backward(&mut self, d: Tensor, _: f32) -> Result<Tensor, DimError> {
        d.reshape(Dim::<Dyn>::new(&self.0)?)
    }
}

struct Dense {
    weights: Ndarr<f32, U2>,
    bias: Ndarr<f32, U1>,
    input: Option<Ndarr<f32, U2>>,
}
impl Dense {
    fn new(nin: usize, nout: usize, rng: &mut Rng) -> Self {
        let scale = (2.0 / nin as f32).sqrt();
        Self {
            weights: Ndarr::from_fn([nin, nout], |_| rng.normal() * scale),
            bias: Ndarr::zeros([nout]),
            input: None,
        }
    }
}
impl Layer for Dense {
    fn forward(&mut self, x: Tensor) -> Result<Tensor, DimError> {
        let x = x.into_ranked::<U2>()?;
        let out = x.mat_mul(&self.weights)? + &self.bias;
        self.input = Some(x);
        Ok(out.into_dyn())
    }
    fn backward(&mut self, gradient: Tensor, lr: f32) -> Result<Tensor, DimError> {
        let x = self.input.take().expect("forward before backward");
        let d = gradient.into_ranked::<U2>()?;
        let dw = x.t_view().mat_mul(&d)?;
        let db = d.reduce(0, f32::add)?;
        let dx = d.mat_mul(&self.weights.t_view())?;
        self.weights.zip_with_in_place(&dw, |w, g| w - lr * g)?;
        self.bias.zip_with_in_place(&db, |b, g| b - lr * g)?;
        Ok(dx.into_dyn())
    }
    fn n_params(&self) -> usize {
        self.weights.len() + self.bias.len()
    }
}

/// Stable mean cross-entropy of [N, classes] logits. Average the gradient over
/// the batch exactly once.
fn softmax_ce(logits: &Tensor, labels: &[usize]) -> Result<(f32, Tensor), DimError> {
    assert_eq!(logits.shape()[0], labels.len());
    assert!(!labels.is_empty());
    let n = labels.len() as f32;
    let onehot = Ndarr::from_fn(logits.dim().clone(), |ix| f32::from(labels[ix[0]] == ix[1]));
    let shifted = logits - &logits.reduce(Keep(1), f32::max)?;
    let exp = shifted.exp();
    let sum = exp.reduce(Keep(1), f32::add)?;
    // Per sample: ln Σ exp - shifted[label]. The gradient is softmax - onehot.
    let picked = (&shifted * &onehot).reduce(Keep(1), f32::add)?;
    let loss = (sum.ln() - picked).iter_elems().sum::<f32>() / n;
    Ok((loss, (&exp / &sum - onehot) / n))
}

struct AlexNet {
    layers: Vec<(&'static str, Box<dyn Layer>)>,
}
impl AlexNet {
    fn new(full: bool, classes: usize, rng: &mut Rng) -> Self {
        let channels = if full {
            [96, 256, 384, 384, 256]
        } else {
            [4, 6, 8, 8, 6]
        };
        let hidden = if full { 4096 } else { 16 };
        let mut side = if full { 227 } else { 67 };
        let mut layers: Vec<(&'static str, Box<dyn Layer>)> = Vec::new();
        let mut cin = 3;
        for (i, (k, stride, padding)) in [(11, 4, 0), (5, 1, 2), (3, 1, 1), (3, 1, 1), (3, 1, 1)]
            .into_iter()
            .enumerate()
        {
            layers.push((
                ["conv1", "conv2", "conv3", "conv4", "conv5"][i],
                Box::new(Conv2D::new(cin, channels[i], k, stride, padding, rng)),
            ));
            layers.push((
                ["relu1", "relu2", "relu3", "relu4", "relu5"][i],
                Box::<ReLU>::default(),
            ));
            side = (side + 2 * padding - k) / stride + 1;
            if matches!(i, 0 | 1 | 4) {
                layers.push((
                    ["pool1", "pool2", "", "", "pool3"][i],
                    Box::new(MaxPool2D::new(3, 2)),
                ));
                side = (side - 3) / 2 + 1;
            }
            cin = channels[i];
        }
        layers.push(("flat", Box::<Flatten>::default()));
        layers.push(("fc6", Box::new(Dense::new(cin * side * side, hidden, rng))));
        layers.push(("relu6", Box::<ReLU>::default()));
        layers.push(("fc7", Box::new(Dense::new(hidden, hidden, rng))));
        layers.push(("relu7", Box::<ReLU>::default()));
        layers.push(("fc8", Box::new(Dense::new(hidden, classes, rng))));
        Self { layers }
    }
    fn forward(&mut self, mut x: Tensor, verbose: bool) -> Result<Tensor, DimError> {
        for (name, layer) in &mut self.layers {
            x = layer.forward(x)?;
            if verbose {
                println!("  {name:6} -> {:?}", x.shape());
            }
        }
        Ok(x)
    }
    fn fit(&mut self, x: &Tensor, labels: &[usize], steps: usize, lr: f32) -> Result<(), DimError> {
        for step in 0..steps {
            let (loss, mut gradient) = softmax_ce(&self.forward(x.clone(), false)?, labels)?;
            for (_, layer) in self.layers.iter_mut().rev() {
                gradient = layer.backward(gradient, lr)?;
            }
            if step % 10 == 0 {
                println!("  step {step:3}: loss {loss:.5}");
            }
        }
        Ok(())
    }
    fn predict(&mut self, x: Tensor) -> Result<Vec<usize>, DimError> {
        let classes = self
            .forward(x, false)?
            .map_cells(U1::new(), |row| argmax(row.iter_elems()).0)?;
        Ok(classes.into_data())
    }
    fn n_params(&self) -> usize {
        self.layers.iter().map(|(_, l)| l.n_params()).sum()
    }
}

/// Four synthetic images: class 0 has a vertical cue, class 1 a horizontal cue.
fn stripes(side: usize, rng: &mut Rng) -> Result<(Tensor, [usize; 4]), DimError> {
    let labels = [0, 1, 0, 1];
    let mut x = Ndarr::from_fn([4, 3, side, side], |_| rng.normal() * 0.1);
    let lo = side * 44 / 100;
    let hi = side * 56 / 100;
    for (n, label) in labels.iter().enumerate() {
        let mut image = x.slice_mut(s![n, .., .., ..])?;
        let mut stripe = if *label == 0 {
            image.slice_mut(s![.., .., lo..hi])?
        } else {
            image.slice_mut(s![.., lo..hi, ..])?
        };
        stripe += 1.0;
    }
    Ok((x.into_dyn(), labels))
}

fn main() -> Result<(), DimError> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let full = args.iter().any(|a| a == "--full");
    let mut steps = if full { 0 } else { 80 };
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--full" => {}
            "--steps" => {
                i += 1;
                steps = args
                    .get(i)
                    .expect("--steps requires a count")
                    .parse()
                    .expect("invalid step count");
            }
            _ => panic!("usage: alexnet [--full] [--steps N]"),
        }
        i += 1;
    }
    let mut rng = Rng(0xC0FFEE);
    let mut net = AlexNet::new(full, if full { 1000 } else { 2 }, &mut rng);
    let side = if full { 227 } else { 67 };
    println!(
        "{} AlexNet-style network: {} parameters",
        if full { "Full" } else { "Compact" },
        net.n_params()
    );
    let x = Ndarr::from_fn([1, 3, side, side], |_| rng.normal() * 0.1).into_dyn();
    println!("  input  -> {:?}", x.shape());
    let out = net.forward(x, true)?;
    println!("Output shape: {:?}", out.shape());
    if steps == 0 {
        return Ok(());
    }

    let (x, labels) = stripes(side, &mut rng)?;
    let start = softmax_ce(&net.forward(x.clone(), false)?, &labels)?.0;
    net.fit(&x, &labels, steps, 0.03)?;
    let end = softmax_ce(&net.forward(x.clone(), false)?, &labels)?.0;
    let predictions = net.predict(x)?;
    let accuracy = predictions
        .iter()
        .zip(labels)
        .filter(|(a, b)| **a == *b)
        .count() as f32
        / labels.len() as f32;
    println!(
        "Loss: {start:.5} -> {end:.5} (5x decrease: {})",
        end < start / 5.0
    );
    println!(
        "Predictions: {predictions:?}; train accuracy: {:.0}%",
        accuracy * 100.0
    );
    Ok(())
}

#[cfg(test)]
#[path = "alexnet/tests.rs"]
mod tests;
