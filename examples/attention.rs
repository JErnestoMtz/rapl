//! Pre-norm causal multi-head self-attention (the attention half of a GPT
//! block), composed from rapl primitives. Inputs are deterministic, so the
//! printed checksum is reproducible. Run with `cargo run --example attention`.

use rapl::*;
use std::ops::Add;

type Tokens = Ndarr<f32, U3>; // [batch, time, model]
type Heads = Ndarr<f32, U4>; // [batch, head, time, head_dim]

/// Deterministic `sin(0.7 i + seed) * scale` in C order, shared by every version.
fn wave<R: Rank>(shape: impl Into<Dim<R>>, seed: f64, scale: f64) -> Ndarr<f32, R> {
    let mut i = 0.0;
    Ndarr::from_fn(shape, |_| {
        let value = ((0.7 * i + seed).sin() * scale) as f32;
        i += 1.0;
        value
    })
}

struct SelfAttention {
    heads: usize,
    ln_gain: Ndarr<f32, U1>,
    ln_bias: Ndarr<f32, U1>,
    wq: Ndarr<f32, U2>,
    bq: Ndarr<f32, U1>,
    wk: Ndarr<f32, U2>,
    bk: Ndarr<f32, U1>,
    wv: Ndarr<f32, U2>,
    bv: Ndarr<f32, U1>,
    wo: Ndarr<f32, U2>,
    bo: Ndarr<f32, U1>,
}

impl SelfAttention {
    fn new(d: usize, heads: usize) -> Self {
        Self {
            heads,
            ln_gain: wave([d], 9.0, 0.1) + 1.0,
            ln_bias: wave([d], 10.0, 0.1),
            wq: wave([d, d], 1.0, 0.3),
            bq: wave([d], 5.0, 0.1),
            wk: wave([d, d], 2.0, 0.3),
            bk: wave([d], 6.0, 0.1),
            wv: wave([d, d], 3.0, 0.3),
            bv: wave([d], 7.0, 0.1),
            wo: wave([d, d], 4.0, 0.3),
            bo: wave([d], 8.0, 0.1),
        }
    }
}

// --- the block ---------------------------------------------------------------

/// Normalize the model axis. `Keep(2)` keeps the reduced axis with extent 1,
/// so each statistic broadcasts back over the row it summarizes.
fn layer_norm(
    x: &Tokens,
    gain: &Ndarr<f32, U1>,
    bias: &Ndarr<f32, U1>,
) -> Result<Tokens, DimError> {
    let n = x.shape()[2] as f32;
    let centered = x - &(x.reduce(Keep(2), f32::add)? / n);
    let var = centered.map(|d| d * d).reduce(Keep(2), f32::add)? / n;
    Ok(&centered / &var.map(|v| (v + 1e-5).sqrt()) * gain + bias)
}

/// [B, T, D] -> [B, H, T, Dh], reusing the projection's storage.
fn split_heads(
    h: &Tokens,
    w: &Ndarr<f32, U2>,
    b: &Ndarr<f32, U1>,
    heads: usize,
) -> Result<Heads, DimError> {
    let [batch, time, model] = h.dim().to_array();
    (h.mat_mul(w)? + b)
        .reshape([batch, time, heads, model / heads])?
        .permute_axes(&[0, 2, 1, 3])
}

impl SelfAttention {
    fn forward(&self, x: &Tokens) -> Result<Tokens, DimError> {
        let [batch, time, model] = x.dim().to_array();
        let head_dim = model / self.heads;
        let h = layer_norm(x, &self.ln_gain, &self.ln_bias)?;
        let q = split_heads(&h, &self.wq, &self.bq, self.heads)?;
        let k = split_heads(&h, &self.wk, &self.bk, self.heads)?;
        let v = split_heads(&h, &self.wv, &self.bv, self.heads)?;

        // Additive mask, broadcast over batch and heads: -inf hides the future.
        let hide = f32::NEG_INFINITY;
        let causal = Ndarr::from_fn([time, time], |ix| if ix[1] > ix[0] { hide } else { 0.0 });
        let k_t = k.permute_axes(&[0, 1, 3, 2])?;
        let scores = q.mat_mul(&k_t)? / (head_dim as f32).sqrt() + &causal; // [B, H, T, T]

        // Softmax along the key axis, as NumPy's `max(-1, keepdims=True)`.
        let exp = (&scores - &scores.reduce(Keep(3), f32::max)?).exp();
        let weights = &exp / &exp.reduce(Keep(3), f32::add)?;

        let merged = weights
            .mat_mul(&v)? // merge heads
            .permute_axes(&[0, 2, 1, 3])?
            .to_owned_array()
            .reshape([batch, time, model])?;
        Ok(x + &(merged.mat_mul(&self.wo)? + &self.bo))
    }
}

// -----------------------------------------------------------------------------

fn main() -> Result<(), DimError> {
    let (batch, time, model, heads) = (2, 5, 8, 2);
    let x = wave([batch, time, model], 0.1, 1.0);
    let y = SelfAttention::new(model, heads).forward(&x)?;
    let checksum: f64 = y.iter_elems().map(|&v| v as f64).sum();
    println!("output {:?}, checksum {checksum:.6}", y.shape());
    let last: Vec<String> = y
        .slice(s![1, -1, ..])?
        .iter_elems()
        .map(|v| format!("{v:.6}"))
        .collect();
    println!("batch 1, last token: [{}]", last.join(", "));
    Ok(())
}

#[cfg(test)]
#[path = "attention/tests.rs"]
mod tests;
