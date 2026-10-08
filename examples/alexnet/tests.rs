use super::*;

const EPS: f32 = 0.002;

fn close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - e).abs() < 0.003 * (1.0 + e.abs()),
            "element {i}: {a} != {e}"
        );
    }
}

// Accumulate the scalar test objective in f64 to reduce cancellation in finite differences.
fn objective(layer: &mut impl Layer, x: &Tensor, upstream: &Tensor) -> f64 {
    layer
        .forward(x.clone())
        .unwrap()
        .iter_elems()
        .zip(upstream.iter_elems())
        .map(|(&a, &b)| a as f64 * b as f64)
        .sum()
}

fn input_gradient(layer: &mut impl Layer, x: &Tensor, d: &Tensor) -> Vec<f32> {
    (0..x.len())
        .map(|i| {
            let mut changed = x.clone();
            changed.data_mut()[i] += EPS;
            let plus = objective(layer, &changed, d);
            changed.data_mut()[i] -= 2.0 * EPS;
            let minus = objective(layer, &changed, d);
            ((plus - minus) / (2.0 * EPS as f64)) as f32
        })
        .collect()
}

#[test]
fn convolution_matches_independent_nchw_reference_and_finite_differences() {
    for (shape, k, stride, padding) in [([2, 2, 3, 4], 2, 1, 0), ([2, 2, 4, 5], 3, 2, 1)] {
        let mut rng = Rng(123);
        let mut conv = Conv2D::new(shape[1], 3, k, stride, padding, &mut rng);
        let x = Ndarr::from_fn(shape, |_| rng.normal() * 0.2).into_dyn();
        let out = conv.forward(x.clone()).unwrap();
        assert_eq!(
            out.shape(),
            [
                shape[0],
                3,
                (shape[2] + 2 * padding - k) / stride + 1,
                (shape[3] + 2 * padding - k) / stride + 1
            ]
        );
        let [n, filters, oh, ow] = out.shape().try_into().unwrap();
        let mut expected = Vec::new();
        for batch in 0..n {
            for filter in 0..filters {
                for y in 0..oh {
                    for z in 0..ow {
                        let mut value = conv.bias[[filter]];
                        for c in 0..shape[1] {
                            for i in 0..k {
                                for j in 0..k {
                                    let iy = (y * stride + i) as isize - padding as isize;
                                    let iz = (z * stride + j) as isize - padding as isize;
                                    if iy >= 0
                                        && iz >= 0
                                        && iy < shape[2] as isize
                                        && iz < shape[3] as isize
                                    {
                                        let index = ((batch * shape[1] + c) * shape[2]
                                            + iy as usize)
                                            * shape[3]
                                            + iz as usize;
                                        value += x.data()[index] * conv.weights[[c, i, j, filter]];
                                    }
                                }
                            }
                        }
                        expected.push(value);
                    }
                }
            }
        }
        // Convolution output keeps its permuted storage; read it in logical order.
        close(out.to_owned_array().data(), &expected);
        let d = Ndarr::from_fn(out.dim().clone(), |_| rng.normal() * 0.1);
        let weights = conv.weights.clone();
        let bias = conv.bias.clone();
        let dx = conv.backward(d.clone(), 1.0).unwrap();
        assert_eq!(dx.shape(), shape);
        let dw = &weights - &conv.weights;
        let db = &bias - &conv.bias;
        conv.weights = weights;
        conv.bias = bias;
        close(dx.data(), &input_gradient(&mut conv, &x, &d));
        for i in 0..conv.weights.len() {
            let original = conv.weights.data()[i];
            conv.weights.data_mut()[i] = original + EPS;
            let plus = objective(&mut conv, &x, &d);
            conv.weights.data_mut()[i] = original - EPS;
            let minus = objective(&mut conv, &x, &d);
            conv.weights.data_mut()[i] = original;
            close(
                &[dw.data()[i]],
                &[((plus - minus) / (2.0 * EPS as f64)) as f32],
            );
        }
        for i in 0..conv.bias.len() {
            conv.bias[[i]] += EPS;
            let plus = objective(&mut conv, &x, &d);
            conv.bias[[i]] -= 2.0 * EPS;
            let minus = objective(&mut conv, &x, &d);
            conv.bias[[i]] += EPS;
            close(&[db[[i]]], &[((plus - minus) / (2.0 * EPS as f64)) as f32]);
        }
    }
}

#[test]
fn dense_gradients_include_exactly_the_upstream_scaling() {
    let mut rng = Rng(456);
    let mut dense = Dense::new(3, 2, &mut rng);
    let x = Ndarr::from([[0.1, -0.3, 0.4], [0.5, 0.2, -0.1]]).into_dyn();
    let d = Ndarr::from([[0.2, -0.1], [-0.3, 0.4]]).into_dyn();
    dense.forward(x.clone()).unwrap();
    let weights = dense.weights.clone();
    let bias = dense.bias.clone();
    let dx = dense.backward(d.clone(), 1.0).unwrap();
    let dw = &weights - &dense.weights;
    let db = &bias - &dense.bias;
    dense.weights = weights;
    dense.bias = bias;
    close(dx.data(), &input_gradient(&mut dense, &x, &d));
    for i in 0..dense.weights.len() {
        let original = dense.weights.data()[i];
        dense.weights.data_mut()[i] = original + EPS;
        let plus = objective(&mut dense, &x, &d);
        dense.weights.data_mut()[i] = original - EPS;
        let minus = objective(&mut dense, &x, &d);
        dense.weights.data_mut()[i] = original;
        close(
            &[dw.data()[i]],
            &[((plus - minus) / (2.0 * EPS as f64)) as f32],
        );
    }
    close(db.data(), &[-0.1, 0.3]);
}

#[test]
fn pooling_accumulates_overlaps_and_chooses_first_tie() {
    let mut pool = MaxPool2D::new(3, 2);
    let mut x = Ndarr::zeros([1, 1, 5, 5]);
    x[[0, 0, 2, 2]] = 9.0;
    let y = pool.forward(x.into_dyn()).unwrap();
    assert_eq!(y.shape(), [1, 1, 2, 2]);
    close(y.data(), &[9.0; 4]);
    let dx = pool.backward(Ndarr::ones(y.dim().clone()), 0.0).unwrap();
    let mut expected = vec![0.0; 25];
    expected[12] = 4.0;
    close(dx.data(), &expected);

    let mut ties = MaxPool2D::new(2, 1);
    let out = ties.forward(Ndarr::ones([1, 1, 3, 3]).into_dyn()).unwrap();
    close(out.data(), &[1.0; 4]);
    let dx = ties.backward(Ndarr::ones(out.dim().clone()), 0.0).unwrap();
    close(dx.data(), &[1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]);

    let x = Ndarr::from(0..40)
        .map(|v| *v as f32 / 10.0)
        .reshape([2, 1, 4, 5])
        .unwrap()
        .into_dyn();
    let out = pool.forward(x.clone()).unwrap();
    let d = Ndarr::fill(0.7, out.dim().clone());
    let dx = pool.backward(d.clone(), 0.0).unwrap();
    close(dx.data(), &input_gradient(&mut pool, &x, &d));
}

#[test]
fn relu_and_flatten_restore_shape_and_gradient() {
    let x = Ndarr::from([-1.0, 0.0, 2.0, -3.0, 4.0, 5.0, -6.0, 7.0])
        .reshape([2, 1, 2, 2])
        .unwrap()
        .into_dyn();
    let mut relu = ReLU::default();
    let mut flat = Flatten::default();
    let out = flat.forward(relu.forward(x).unwrap()).unwrap();
    assert_eq!(out.shape(), [2, 4]);
    close(out.data(), &[0.0, 0.0, 2.0, 0.0, 4.0, 5.0, 0.0, 7.0]);
    let d = flat.backward(Ndarr::ones(out.dim().clone()), 0.0).unwrap();
    let dx = relu.backward(d, 0.0).unwrap();
    assert_eq!(dx.shape(), [2, 1, 2, 2]);
    close(dx.data(), &[0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]);
}

#[test]
fn mean_cross_entropy_has_stable_correct_gradient() {
    let x = Ndarr::from([[1.0, -2.0, 0.5], [-1.0, 2.0, 0.0]]).into_dyn();
    let labels = [0, 2];
    let (loss, gradient) = softmax_ce(&x, &labels).unwrap();
    for i in 0..x.len() {
        let mut changed = x.clone();
        changed.data_mut()[i] += EPS;
        let plus = softmax_ce(&changed, &labels).unwrap().0;
        changed.data_mut()[i] -= 2.0 * EPS;
        let minus = softmax_ce(&changed, &labels).unwrap().0;
        close(&[gradient.data()[i]], &[(plus - minus) / (2.0 * EPS)]);
    }
    let row_sums = gradient.reduce(1, f32::add).unwrap();
    assert!(row_sums.iter_elems().all(|s| s.abs() < 1e-6));
    let (shifted_loss, shifted_gradient) = softmax_ce(&(&x + 10_000.0), &labels).unwrap();
    close(&[shifted_loss], &[loss]);
    close(shifted_gradient.data(), gradient.data());
    let extreme = Ndarr::from([[10_000.0, -10_000.0]]).into_dyn();
    assert_eq!(softmax_ce(&extreme, &[1]).unwrap().0, 20_000.0);
}

#[test]
fn compact_network_has_all_nineteen_layers_and_expected_shapes() {
    let mut net = AlexNet::new(false, 2, &mut Rng(1));
    assert_eq!(net.n_params(), 3942);
    let expected: &[&[usize]] = &[
        &[1, 4, 15, 15],
        &[1, 4, 15, 15],
        &[1, 4, 7, 7],
        &[1, 6, 7, 7],
        &[1, 6, 7, 7],
        &[1, 6, 3, 3],
        &[1, 8, 3, 3],
        &[1, 8, 3, 3],
        &[1, 8, 3, 3],
        &[1, 8, 3, 3],
        &[1, 6, 3, 3],
        &[1, 6, 3, 3],
        &[1, 6, 1, 1],
        &[1, 6],
        &[1, 16],
        &[1, 16],
        &[1, 16],
        &[1, 16],
        &[1, 2],
    ];
    assert_eq!(net.layers.len(), expected.len());
    let mut x = Ndarr::zeros([1, 3, 67, 67]).into_dyn();
    for ((name, layer), shape) in net.layers.iter_mut().zip(expected) {
        x = layer.forward(x).unwrap();
        assert_eq!(x.shape(), *shape, "{name}");
    }
}

#[test]
fn complete_network_learns_the_synthetic_batch() {
    let mut rng = Rng(0xC0FFEE);
    let mut net = AlexNet::new(false, 2, &mut rng);
    // Match the demo's random stream, including its architecture forward pass.
    net.forward(
        Ndarr::from_fn([1, 3, 67, 67], |_| rng.normal() * 0.1).into_dyn(),
        false,
    )
    .unwrap();
    let (x, labels) = stripes(67, &mut rng).unwrap();
    let start = softmax_ce(&net.forward(x.clone(), false).unwrap(), &labels)
        .unwrap()
        .0;
    net.fit(&x, &labels, 20, 0.03).unwrap();
    let end = softmax_ce(&net.forward(x.clone(), false).unwrap(), &labels)
        .unwrap()
        .0;
    assert!(end < start / 5.0, "loss {start} -> {end}");
    assert_eq!(net.predict(x).unwrap(), labels);
}
