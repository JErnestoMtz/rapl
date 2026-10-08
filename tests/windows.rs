//! `Win(w)` window-selector tests: `windows[p, k] == input[p + k]`, checked against a brute-force oracle.

use rapl::{s, Dim, NdView, Ndarr, SliceSpec, Win, U1, U2, U3};

/// Row-major flat index of `idx` in `shape`.
fn flat(shape: &[usize], idx: &[usize]) -> usize {
    idx.iter().zip(shape).fold(0, |acc, (i, s)| acc * s + i)
}

/// Visits every multi-index of `shape` in row-major order.
fn for_each_index(shape: &[usize], mut f: impl FnMut(&[usize])) {
    let n: usize = shape.iter().product();
    let mut idx = vec![0; shape.len()];
    for _ in 0..n {
        f(&idx);
        for ax in (0..shape.len()).rev() {
            idx[ax] += 1;
            if idx[ax] < shape[ax] {
                break;
            }
            idx[ax] = 0;
        }
    }
}

/// Brute-force windows: output splices `[n - w + 1, w]` at `axis`; element (p, k) reads `input[.., p + k, ..]`.
fn ref_windows(shape: &[usize], data: &[i32], axis: usize, w: usize) -> (Vec<usize>, Vec<i32>) {
    let mut out_shape = shape.to_vec();
    out_shape[axis] = shape[axis] - w + 1;
    out_shape.insert(axis + 1, w);
    let mut out = Vec::new();
    for_each_index(&out_shape, |oidx| {
        let mut iidx = Vec::with_capacity(shape.len());
        iidx.extend_from_slice(&oidx[..axis]);
        iidx.push(oidx[axis] + oidx[axis + 1]);
        iidx.extend_from_slice(&oidx[axis + 2..]);
        out.push(data[flat(shape, &iidx)]);
    });
    (out_shape, out)
}

/// Windows of a rank-3 view along `axis`, materialized as (shape, elements).
fn win_elems(a: &NdView<'_, i32, U3>, axis: usize, w: usize) -> (Vec<usize>, Vec<i32>) {
    fn collect<'a>(v: &rapl::NdView<'a, i32, rapl::U4>) -> (Vec<usize>, Vec<i32>) {
        (v.shape().to_vec(), v.iter_elems().cloned().collect())
    }
    match axis {
        0 => collect(&a.slice(s![Win(w), .., ..]).unwrap()),
        1 => collect(&a.slice(s![.., Win(w), ..]).unwrap()),
        2 => collect(&a.slice(s![.., .., Win(w)]).unwrap()),
        _ => unreachable!(),
    }
}

fn seq(n: usize) -> Vec<i32> {
    (0..n as i32).collect()
}

/// Every axis and width 0..=extent, on contiguous and non-contiguous layouts (the oracle runs on logical elements).
#[test]
#[allow(clippy::reversed_empty_ranges)] // `1..-1` is the grammar's negative end, not a std range
fn oracle_every_axis_and_layout() {
    let base = Ndarr::new(&seq(3 * 4 * 5), [3, 4, 5]).unwrap();
    let variants: Vec<(&str, NdView<'_, i32, U3>)> = vec![
        ("owned", base.view()),
        ("transposed", base.view().permute_axes(&[2, 0, 1]).unwrap()),
        ("negative step", base.slice(s![.., ..;-1, ..]).unwrap()),
        ("offset", base.slice(s![1.., .., 1..-1]).unwrap()),
    ];
    for (name, view) in &variants {
        let shape = view.shape().to_vec();
        let elems: Vec<i32> = view.iter_elems().cloned().collect();
        for axis in 0..3 {
            for w in 0..=shape[axis] {
                let (got_shape, got) = win_elems(view, axis, w);
                let (want_shape, want) = ref_windows(&shape, &elems, axis, w);
                assert_eq!(got_shape, want_shape, "{name}, axis {axis}, w {w}");
                assert_eq!(got, want, "{name}, axis {axis}, w {w}");
            }
        }
    }
}

/// Windows of a windows view: `v2[p, q, k] == input[p + q + k]`.
#[test]
fn windows_compose_with_windows() {
    let series = Ndarr::from([0, 1, 2, 3, 4, 5]);
    let doubled = series.slice(s![Win(4)]).unwrap();
    let nested = doubled.slice(s![.., Win(2)]).unwrap();
    assert_eq!(nested.shape(), &[3, 3, 2]);
    let got: Vec<i32> = nested.iter_elems().cloned().collect();
    let mut want = Vec::new();
    for_each_index(&[3, 3, 2], |idx| {
        want.push((idx[0] + idx[1] + idx[2]) as i32)
    });
    assert_eq!(got, want);
}

/// Window stride is a step on the position axis, after `Win`.
#[test]
fn window_stride_is_position_step() {
    let series = Ndarr::new(&seq(10), [10]).unwrap();
    let windows = series.slice(s![Win(4)]).unwrap();
    let strided = windows.slice(s![..;2, ..]).unwrap();
    assert_eq!(strided.shape(), &[4, 4]);
    let got: Vec<i32> = strided.iter_elems().cloned().collect();
    let mut want = Vec::new();
    for p in [0usize, 2, 4, 6] {
        want.extend((0..4).map(|k| (p + k) as i32));
    }
    assert_eq!(got, want);
}

/// Window dilation is a step on the window axis, after `Win`.
#[test]
fn window_dilation_is_window_step() {
    let series = Ndarr::new(&seq(10), [10]).unwrap();
    let windows = series.slice(s![Win(5)]).unwrap();
    let dilated = windows.slice(s![.., ..;2]).unwrap();
    assert_eq!(dilated.shape(), &[6, 3]);
    let got: Vec<i32> = dilated.iter_elems().cloned().collect();
    let mut want = Vec::new();
    for p in 0..6usize {
        want.extend([p, p + 2, p + 4].map(|i| i as i32));
    }
    assert_eq!(got, want);
}

/// A bounded window is a slice before `Win`.
#[test]
fn bounded_window_is_slice_then_win() {
    let series = Ndarr::new(&seq(10), [10]).unwrap();
    let sub = series.slice(s![2..8]).unwrap();
    let bounded = sub.slice(s![Win(3)]).unwrap();
    let (want_shape, want) = ref_windows(&[6], &seq(10)[2..8], 0, 3);
    assert_eq!(bounded.shape(), &want_shape[..]);
    assert_eq!(bounded.iter_elems().cloned().collect::<Vec<_>>(), want);
}

/// Boxcar moving sum: `fold_axis` over the window axis (two axes sharing one stride).
#[test]
fn fold_axis_over_window_axis() {
    let series = Ndarr::from([1, 2, 3, 4, 5]);
    let sums = series
        .slice(s![Win(3)])
        .unwrap()
        .fold_axis(1, 0, |acc, x| acc + x)
        .unwrap();
    assert_eq!(sums, Ndarr::from([6, 9, 12]));
}

/// Moving maximum: `reduce` over the window axis.
#[test]
fn reduce_over_window_axis() {
    let series = Ndarr::from([3, 1, 4, 1, 5, 9, 2, 6]);
    let max = series
        .slice(s![Win(3)])
        .unwrap()
        .reduce(1, i32::max)
        .unwrap();
    assert_eq!(max, Ndarr::from([4, 4, 5, 9, 9, 9]));
}

/// FIR per channel: windows view + one permute + fused contraction, O(output) memory.
#[test]
fn contract_over_windows_view_is_fir() {
    let spec = Ndarr::new(&seq(6 * 2), [6, 2]).unwrap(); // [t, f]
    let kernel = Ndarr::from([1, -2, 3]);
    let w = kernel.shape()[0];
    let raw = spec.slice(s![Win(w), ..]).unwrap(); // [t', w, f]
    let windows = raw.view().permute_axes(&[0, 2, 1]).unwrap(); // [t', f, w]
    let got = windows
        .contract(&kernel, U1::default(), |x, k| x * k, |a, b| a + b)
        .unwrap();
    assert_eq!(got.shape(), &[4, 2]);
    let mut want = Vec::new();
    for t in 0..4i32 {
        for f in 0..2i32 {
            let sample = |k: i32| (t + k) * 2 + f;
            want.push(sample(0) - 2 * sample(1) + 3 * sample(2));
        }
    }
    assert_eq!(got.iter_elems().cloned().collect::<Vec<_>>(), want);
}

/// Regrouping window axes is `permute_axes_view`.
#[test]
fn permute_regroups_window_axes() {
    let image = Ndarr::new(&seq(4 * 5 * 2), [4, 5, 2]).unwrap();
    let raw = image.slice(s![Win(2), Win(3), ..]).unwrap(); // [3, 2, 3, 3, 2]
    let patches = raw.view().permute_axes(&[0, 2, 1, 3, 4]).unwrap(); // [3, 3, 2, 3, 2]
    assert_eq!(patches.shape(), &[3, 3, 2, 3, 2]);
    let got: Vec<i32> = patches.iter_elems().cloned().collect();
    let mut want = Vec::new();
    for_each_index(&[3, 3, 2, 3, 2], |idx| {
        let (h, w, kh, kw, c) = (idx[0], idx[1], idx[2], idx[3], idx[4]);
        want.push(flat(&[4, 5, 2], &[h + kh, w + kw, c]) as i32);
    });
    assert_eq!(got, want);
}

/// `Win` contributes two output axes to the compile-time rank fold.
#[test]
#[allow(clippy::reversed_empty_ranges)] // `1..-1` is the grammar's negative end, not a std range
fn typed_output_ranks() {
    let series = Ndarr::from([1, 2, 3, 4]);
    let windows: NdView<'_, i32, U2> = series.slice(s![Win(2)]).unwrap();
    assert_eq!(windows.shape(), &[3, 2]);

    let cube = Ndarr::new(&seq(3 * 4 * 5), [3, 4, 5]).unwrap();
    // U0 (index) + U2 (Win) + U1 (range) = U3.
    let mixed: NdView<'_, i32, U3> = cube.slice(s![2, Win(4), 1..-1]).unwrap();
    assert_eq!(mixed.shape(), &[1, 4, 3]);
}

/// `SliceSpec::Win` via `&[SliceSpec]`: runtime axis position, yields `Dyn`.
#[test]
fn runtime_escape_hatch() {
    let a = Ndarr::from([[1, 2, 3], [4, 5, 6]]);
    for axis in 0..2 {
        let mut specs = [SliceSpec::from(..); 2];
        specs[axis] = SliceSpec::Win(2);
        let view = a.slice(&specs[..]).unwrap();
        let (want_shape, want) = ref_windows(&[2, 3], &[1, 2, 3, 4, 5, 6], axis, 2);
        assert_eq!(view.shape(), &want_shape[..]);
        assert_eq!(view.iter_elems().cloned().collect::<Vec<_>>(), want);
    }
}

#[test]
fn bounds() {
    let series = Ndarr::from([1, 2, 3, 4]);
    assert!(series.slice(s![Win(5)]).is_err());
    let full = series.slice(s![Win(4)]).unwrap();
    assert_eq!(full.shape(), &[1, 4]);
    assert_eq!(
        full.iter_elems().cloned().collect::<Vec<_>>(),
        vec![1, 2, 3, 4]
    );
    // w = 0 yields n + 1 empty windows, consistent with zero-extent axes.
    let empty = series.slice(s![Win(0)]).unwrap();
    assert_eq!(empty.shape(), &[5, 0]);
    assert_eq!(empty.len(), 0);
    // An empty axis admits only w = 0 (one empty window); w = 1 exceeds it.
    let none: Ndarr<i32, U1> = Ndarr::from_vec_dim(vec![], Dim::new(&[0]).unwrap()).unwrap();
    assert_eq!(none.slice(s![Win(0)]).unwrap().shape(), &[1, 0]);
    assert!(none.slice(s![Win(1)]).is_err());
}

/// Windows alias one element into many windows, so mutable selection rejects them.
#[test]
fn slice_mut_rejects_win() {
    let mut a = Ndarr::from([[1, 2, 3], [4, 5, 6]]);
    assert!(a.slice_mut(s![Win(2), ..]).is_err());
    let specs = [SliceSpec::Win(2), SliceSpec::from(..)];
    assert!(a.slice_mut(&specs[..]).is_err());
    // The same selection is fine read-only.
    assert!(a.slice(s![Win(2), ..]).is_ok());
}

/// 2x2 max pooling, stride 2: `Win` + position steps + `reduce` — no pooling kernel.
#[test]
fn max_pool_is_win_plus_step_plus_reduce() {
    let image = Ndarr::new(&seq(4 * 6 * 2), [4, 6, 2]).unwrap();
    let patches = image.slice(s![Win(2), Win(2), ..]).unwrap(); // [3, 2, 5, 2, 2]
    let pooled = patches
        .slice(s![..;2, .., ..;2, .., ..])
        .unwrap() // [2, 2, 3, 2, 2]
        .reduce(3, i32::max)
        .unwrap() // [2, 2, 3, 2]
        .reduce(1, i32::max)
        .unwrap(); // [2, 3, 2]
    assert_eq!(pooled.shape(), &[2, 3, 2]);
    let got: Vec<i32> = pooled.iter_elems().cloned().collect();
    let mut want = Vec::new();
    for_each_index(&[2, 3, 2], |idx| {
        let (h, w, c) = (idx[0] * 2, idx[1] * 2, idx[2]);
        let at = |dh: usize, dw: usize| flat(&[4, 6, 2], &[h + dh, w + dw, c]) as i32;
        want.push(at(0, 0).max(at(0, 1)).max(at(1, 0)).max(at(1, 1)));
    });
    assert_eq!(got, want);
}

/// conv2d = windows view + one permute + one fused contraction, checked against brute force.
#[test]
fn conv2d_is_windows_plus_contract() {
    use rapl::{U3 as R3, U4 as R4};

    /// x: [h, w, c_in], k: [c_out, kh, kw, c_in] -> [h', w', c_out]
    fn conv2d(x: &Ndarr<i32, R3>, k: &Ndarr<i32, R4>) -> Ndarr<i32, R3> {
        let (kh, kw) = (k.shape()[1], k.shape()[2]);
        let raw = x.slice(s![Win(kh), Win(kw), ..]).unwrap(); // [h', kh, w', kw, c]
        let patches = raw.view().permute_axes(&[0, 2, 1, 3, 4]).unwrap(); // [h', w', kh, kw, c]
        let filters = k.view().permute_axes(&[1, 2, 3, 0]).unwrap(); // [kh, kw, c, c_out]
        patches
            .contract(&filters, R3::default(), |x, w| x * w, |a, b| a + b)
            .unwrap()
    }

    let (h, w, c_in, c_out, kh, kw) = (5, 6, 3, 2, 2, 3);
    let x = Ndarr::new(&seq(h * w * c_in), [h, w, c_in]).unwrap();
    let kernel_data: Vec<i32> = (0..(c_out * kh * kw * c_in) as i32)
        .map(|i| (i % 5) - 2)
        .collect();
    let k = Ndarr::new(&kernel_data, [c_out, kh, kw, c_in]).unwrap();

    let got = conv2d(&x, &k);
    assert_eq!(got.shape(), &[h - kh + 1, w - kw + 1, c_out]);

    let mut want = Vec::new();
    for_each_index(&[h - kh + 1, w - kw + 1, c_out], |idx| {
        let (oh, ow, oc) = (idx[0], idx[1], idx[2]);
        let mut acc = 0i32;
        for dh in 0..kh {
            for dw in 0..kw {
                for ci in 0..c_in {
                    let xv = flat(&[h, w, c_in], &[oh + dh, ow + dw, ci]) as i32;
                    let kv = kernel_data[flat(&[c_out, kh, kw, c_in], &[oc, dh, dw, ci])];
                    acc += xv * kv;
                }
            }
        }
        want.push(acc);
    });
    assert_eq!(got.iter_elems().cloned().collect::<Vec<_>>(), want);
}
