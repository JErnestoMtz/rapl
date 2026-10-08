use super::*;

/// Independent f64 loop reference that reads inputs only through indexing and
/// applies the causal mask by limiting the key range instead of using -inf.
fn reference(block: &SelfAttention, x: &Tokens) -> Vec<f64> {
    let [batch, time, model] = x.dim().to_array();
    let head_dim = model / block.heads;
    let mut out = Vec::new();
    for b in 0..batch {
        let h: Vec<Vec<f64>> = (0..time)
            .map(|t| {
                let row: Vec<f64> = (0..model).map(|d| x[[b, t, d]] as f64).collect();
                let mean = row.iter().sum::<f64>() / model as f64;
                let var = row.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / model as f64;
                (0..model)
                    .map(|d| {
                        (row[d] - mean) / (var + 1e-5).sqrt() * block.ln_gain[[d]] as f64
                            + block.ln_bias[[d]] as f64
                    })
                    .collect()
            })
            .collect();
        let project = |w: &Ndarr<f32, U2>, bias: &Ndarr<f32, U1>, t: usize, j: usize| {
            bias[[j]] as f64 + (0..model).map(|d| h[t][d] * w[[d, j]] as f64).sum::<f64>()
        };
        let mut merged = vec![vec![0.0; model]; time];
        for head in 0..block.heads {
            let column = |e: usize| head * head_dim + e;
            for i in 0..time {
                let scores: Vec<f64> = (0..=i)
                    .map(|j| {
                        (0..head_dim)
                            .map(|e| {
                                project(&block.wq, &block.bq, i, column(e))
                                    * project(&block.wk, &block.bk, j, column(e))
                            })
                            .sum::<f64>()
                            / (head_dim as f64).sqrt()
                    })
                    .collect();
                let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let total: f64 = scores.iter().map(|s| (s - max).exp()).sum();
                for e in 0..head_dim {
                    merged[i][column(e)] = (0..=i)
                        .map(|j| {
                            (scores[j] - max).exp() / total
                                * project(&block.wv, &block.bv, j, column(e))
                        })
                        .sum();
                }
            }
        }
        for (t, merged) in merged.iter().enumerate() {
            for j in 0..model {
                let projected = block.bo[[j]] as f64
                    + (0..model)
                        .map(|d| merged[d] * block.wo[[d, j]] as f64)
                        .sum::<f64>();
                out.push(x[[b, t, j]] as f64 + projected);
            }
        }
    }
    out
}

#[test]
fn matches_an_independent_loop_reference() {
    for (batch, time, model, heads) in [(2, 5, 8, 2), (3, 4, 12, 3), (1, 1, 4, 1)] {
        let block = SelfAttention::new(model, heads);
        let x = wave([batch, time, model], 0.3, 1.5);
        let y = block.forward(&x).unwrap();
        assert_eq!(y.shape(), [batch, time, model]);
        for (i, (&actual, expected)) in y.data().iter().zip(reference(&block, &x)).enumerate() {
            assert!(
                (actual as f64 - expected).abs() < 1e-5 * (1.0 + expected.abs()),
                "element {i}: {actual} != {expected}"
            );
        }
    }
}

#[test]
fn future_tokens_do_not_change_earlier_outputs() {
    let block = SelfAttention::new(8, 2);
    let x = wave([2, 5, 8], 0.1, 1.0);
    let mut changed = x.clone();
    let mut last_token = changed.slice_mut(s![.., -1, ..]).unwrap();
    last_token += 3.0;
    let (y, y_changed) = (block.forward(&x).unwrap(), block.forward(&changed).unwrap());
    let earlier = s![.., ..-1, ..];
    assert_eq!(y.slice(earlier).unwrap(), y_changed.slice(earlier).unwrap());
    assert_ne!(
        y.slice(s![.., -1, ..]).unwrap(),
        y_changed.slice(s![.., -1, ..]).unwrap()
    );
}
