//! Constructors, spaces, FFT and composed utilities, checked against golden values and reference implementations.

#![allow(clippy::excessive_precision)]

use num_traits::Float;
use rapl::{
    BroadcastRank, Broadcasted, Dim, DimError, Dyn, InsertAxisRank, Inserted, Ndarr, Rank,
    RemoveAxisRank, Removed, U0, U1, U2, U3,
};
use std::fmt::Debug;

/// Owned rank-reduced axis slices; pins `index_axis_view`, including the out-of-bounds panic.
trait EagerSlices<T: Clone, R: Rank + RemoveAxisRank> {
    fn slice_at(&self, axis: usize) -> Vec<Ndarr<T, Removed<R>>>;
}

impl<T: Clone, R: Rank + RemoveAxisRank> EagerSlices<T, R> for Ndarr<T, R> {
    fn slice_at(&self, axis: usize) -> Vec<Ndarr<T, Removed<R>>> {
        assert!(axis < self.rank(), "axis out of bounds");
        (0..self.shape()[axis])
            .map(|index| self.index_axis_view(axis, index).unwrap().to_owned_array())
            .collect()
    }
}

/// Reference `broadcast`/`broadcast_to`: `broadcast_view_to(..).to_owned_array()`; `broadcast` co-broadcasts shapes first.
trait EagerBroadcast<T: Clone, R: Rank> {
    fn broadcast<R2: Rank, D: Into<Dim<R2>>>(
        &self,
        shape: D,
    ) -> Result<Ndarr<T, Broadcasted<R, R2>>, DimError>
    where
        R: BroadcastRank<R2>;
    fn broadcast_to<R2: Rank, D: Into<Dim<R2>>>(&self, shape: D) -> Result<Ndarr<T, R2>, DimError>;
}

impl<T: Clone, R: Rank> EagerBroadcast<T, R> for Ndarr<T, R> {
    fn broadcast<R2: Rank, D: Into<Dim<R2>>>(
        &self,
        shape: D,
    ) -> Result<Ndarr<T, Broadcasted<R, R2>>, DimError>
    where
        R: BroadcastRank<R2>,
    {
        let shape = self.dim().broadcast_shape(&shape.into())?;
        Ok(self.broadcast_view_to(&shape)?.to_owned_array())
    }

    fn broadcast_to<R2: Rank, D: Into<Dim<R2>>>(&self, shape: D) -> Result<Ndarr<T, R2>, DimError> {
        Ok(self.broadcast_view_to(&shape.into())?.to_owned_array())
    }
}

/// Reference `logspace`/`geomspace`: `linspace(..).map(..)`.
fn logspace<T: Float>(start: T, end: T, base: T, n: u16) -> Ndarr<T, U1>
where
    u16: TryInto<T>,
    <u16 as TryInto<T>>::Error: Debug,
{
    Ndarr::linspace(start, end, n).map(|x| base.powf(*x))
}

fn geomspace<T: Float>(start: T, end: T, n: u16) -> Ndarr<T, U1>
where
    u16: TryInto<T>,
    <u16 as TryInto<T>>::Error: Debug,
{
    let ratio = end / start;
    Ndarr::linspace(T::zero(), T::one(), n).map(|t| start * ratio.powf(*t))
}

fn stack<T: Clone, R: Rank + InsertAxisRank>(
    arrays: &[Ndarr<T, R>],
    axis: usize,
) -> Ndarr<T, Inserted<R>> {
    let expanded: Vec<_> = arrays
        .iter()
        .map(|array| array.insert_axis_view(axis).unwrap())
        .collect();
    Ndarr::concatenate(axis, &expanded).unwrap()
}

struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Lcg(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }

    fn f64_range(&mut self, lo: f64, hi: f64) -> f64 {
        let unit = (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
        lo + unit * (hi - lo)
    }

    /// Inclusive range.
    fn usize_range(&mut self, lo: usize, hi: usize) -> usize {
        lo + (self.next_u64() % (hi - lo + 1) as u64) as usize
    }

    /// Inclusive range.
    fn i32_range(&mut self, lo: i32, hi: i32) -> i32 {
        lo + (self.next_u64() % ((hi - lo + 1) as u64)) as i32
    }

    /// Inclusive range.
    fn isize_range(&mut self, lo: isize, hi: isize) -> isize {
        lo + (self.next_u64() % ((hi - lo + 1) as u64)) as isize
    }
}

fn make_f64_lcg(n: usize, g: &mut Lcg) -> Vec<f64> {
    (0..n).map(|_| g.f64_range(-4.0, 4.0)).collect()
}

/// Deterministic integer test data.
fn make_i32(shape: &[usize], start: i32) -> Vec<i32> {
    let n = product(shape);
    (0..n as i32).map(|i| start + i * 7 % 101).collect()
}

fn product(shape: &[usize]) -> usize {
    shape.iter().product()
}

fn assert_eq_slices<T: PartialEq + Debug>(op: &str, got: &[T], want: &[T]) {
    assert_eq!(got.len(), want.len(), "{op}: length mismatch");
    for (i, (a, b)) in got.iter().zip(want.iter()).enumerate() {
        if a != b {
            panic!("{op}: mismatch at flat index {i}: got={a:?} want={b:?}");
        }
    }
}

/// NaN/inf tolerant elementwise compare with a relative-ish epsilon.
fn assert_close<T: Float + Debug>(op: &str, got: &[T], want: &[T], eps: T) {
    assert_eq!(got.len(), want.len(), "{op}: length mismatch");
    for (i, (a, b)) in got.iter().zip(want.iter()).enumerate() {
        let scale = a.abs().max(b.abs()).max(T::one());
        let ok = *a == *b || (a.is_nan() && b.is_nan()) || (*a - *b).abs() <= eps * scale;
        if !ok {
            panic!("{op}: mismatch at flat index {i}: got={a:?} want={b:?}");
        }
    }
}

fn panics<F: FnOnce()>(f: F) -> bool {
    let prev = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let out = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f));
    std::panic::set_hook(prev);
    out.is_err()
}

fn unravel(mut n: usize, shape: &[usize]) -> Vec<usize> {
    let mut idx = vec![0usize; shape.len()];
    for i in (0..shape.len()).rev() {
        let d = shape[i].max(1);
        idx[i] = n % d;
        n /= d;
    }
    idx
}

fn ravel(idx: &[usize], shape: &[usize]) -> usize {
    let mut n = 0usize;
    for (i, &d) in shape.iter().enumerate() {
        n = n * d + idx[i];
    }
    n
}

fn ref_transpose<T: Clone>(data: &[T], shape: &[usize]) -> (Vec<usize>, Vec<T>) {
    let mut tshape = shape.to_vec();
    tshape.reverse();
    let n = product(shape);
    let mut out = Vec::with_capacity(n);
    for j in 0..n {
        let mut idx = unravel(j, &tshape);
        idx.reverse();
        out.push(data[ravel(&idx, shape)].clone());
    }
    (tshape, out)
}

fn ref_slice_at<T: Clone>(data: &[T], shape: &[usize], axis: usize, index: usize) -> Vec<T> {
    let mut sub = shape.to_vec();
    sub.remove(axis);
    (0..product(&sub))
        .map(|n| {
            let mut idx = unravel(n, &sub);
            idx.insert(axis, index);
            data[ravel(&idx, shape)].clone()
        })
        .collect()
}

/// Mirrors `reduce`'s fold order: acc starts as slice 0, then acc = f(acc, next).
fn ref_reduce<T: Clone>(
    data: &[T],
    shape: &[usize],
    axis: usize,
    f: impl Fn(T, T) -> T,
) -> (Vec<usize>, Vec<T>) {
    let mut out_shape = shape.to_vec();
    out_shape.remove(axis);
    let mut out = ref_slice_at(data, shape, axis, 0);
    for i in 1..shape[axis] {
        let next = ref_slice_at(data, shape, axis, i);
        for (acc, value) in out.iter_mut().zip(next) {
            *acc = f(acc.clone(), value);
        }
    }
    (out_shape, out)
}

fn ref_roll<T: Clone>(data: &[T], shape: &[usize], shift: isize, axis: usize) -> Vec<T> {
    let axis_len = shape[axis];
    if axis_len == 0 {
        return data.to_vec();
    }
    let shift = shift.rem_euclid(axis_len as isize) as usize;
    (0..data.len())
        .map(|n| {
            let mut idx = unravel(n, shape);
            idx[axis] = (idx[axis] + axis_len - shift) % axis_len;
            data[ravel(&idx, shape)].clone()
        })
        .collect()
}

/// NumPy-style right-aligned broadcast of `data` (shape `src`) to `target`.
fn ref_broadcast<T: Clone>(data: &[T], src: &[usize], target: &[usize]) -> Vec<T> {
    let off = target.len() - src.len();
    (0..product(target))
        .map(|n| {
            let idx = unravel(n, target);
            let sidx: Vec<usize> = (0..src.len())
                .map(|k| if src[k] == 1 { 0 } else { idx[k + off] })
                .collect();
            data[ravel(&sidx, src)].clone()
        })
        .collect()
}

fn ref_stack<T: Clone>(
    slices: &[Vec<T>],
    slice_shape: &[usize],
    axis: usize,
) -> (Vec<usize>, Vec<T>) {
    let mut out_shape = slice_shape.to_vec();
    out_shape.insert(axis, slices.len());
    let out = (0..product(&out_shape))
        .map(|n| {
            let mut idx = unravel(n, &out_shape);
            let i = idx.remove(axis);
            slices[i][ravel(&idx, slice_shape)].clone()
        })
        .collect();
    (out_shape, out)
}

fn ref_softmax<T: Float>(xs: &[T]) -> Vec<T> {
    let max = xs.iter().cloned().reduce(T::max).unwrap();
    let exp: Vec<T> = xs.iter().map(|&x| (x - max).exp()).collect();
    let sum = exp.iter().cloned().fold(T::zero(), |a, b| a + b);
    exp.into_iter().map(|e| e / sum).collect()
}

const SHAPES_1_UTILS: [[usize; 1]; 4] = [[1], [3], [7], [0]];
const SHAPES_2_UTILS: [[usize; 2]; 7] = [[1, 1], [1, 5], [5, 1], [2, 3], [3, 2], [0, 3], [3, 0]];
const SHAPES_3_UTILS: [[usize; 3]; 5] = [[1, 1, 1], [2, 3, 4], [3, 1, 4], [4, 1, 1], [2, 0, 3]];

const SHAPES_2_API: [[usize; 2]; 6] = [[2, 3], [3, 2], [1, 5], [5, 1], [1, 1], [4, 4]];
const SHAPES_3_API: [[usize; 3]; 6] = [
    [2, 3, 4],
    [3, 1, 4],
    [1, 1, 5],
    [4, 3, 1],
    [1, 5, 1],
    [2, 2, 2],
];

fn rank0(value: i32) -> Ndarr<i32, U0> {
    Ndarr::new(&[value], Dim::<U0>::new(&[]).unwrap()).unwrap()
}

#[test]
fn fill_rank1() {
    for shape in SHAPES_1_UTILS {
        let a = Ndarr::<i32, U1>::zeros(shape);
        assert_eq!(a.shape(), &shape);
        assert!(a.data().iter().all(|v| *v == 0), "zeros_i32 {shape:?}");

        let a = Ndarr::<f64, U1>::ones(shape);
        assert!(a.data().iter().all(|v| *v == 1.0), "ones_f64 {shape:?}");

        let a = Ndarr::<i32, U1>::fill(-9, shape);
        assert!(a.data().iter().all(|v| *v == -9), "fill_i32 {shape:?}");
        assert_eq!(a.data().len(), product(&shape));
    }
}

#[test]
fn fill_rank2_rank3_asymmetric() {
    for shape in SHAPES_2_UTILS {
        let a = Ndarr::<u8, U2>::zeros(shape);
        assert_eq!(a.shape(), &shape);
        assert!(a.data().iter().all(|v| *v == 0), "zeros_u8 {shape:?}");

        let a = Ndarr::<f32, U2>::fill(2.5, shape);
        assert!(a.data().iter().all(|v| *v == 2.5), "fill_f32 {shape:?}");
        assert_eq!(a.data().len(), product(&shape), "fill len {shape:?}");
    }
    for shape in SHAPES_3_UTILS {
        let a = Ndarr::<i64, U3>::ones(shape);
        assert_eq!(a.shape(), &shape);
        assert!(a.data().iter().all(|v| *v == 1), "ones_i64 {shape:?}");
        assert_eq!(a.data().len(), product(&shape));
    }
}

#[test]
fn fill_rank0() {
    let dim = Dim::<U0>::new(&[]).unwrap();
    let a = Ndarr::<f64, U0>::zeros(&dim);
    assert!(a.shape().is_empty());
    assert_eq!(a.data(), &[0.0]);
    assert_eq!(a.data().len(), 1, "rank-0 array holds one element");

    let a = Ndarr::<i32, U0>::fill(42, &dim);
    assert_eq!(a.data(), &[42]);
}

#[test]
fn fill_random_shapes() {
    let mut g = Lcg::new(0x000A_0D17_C0DE_5EED);
    for trial in 0..24 {
        let shape = [
            g.usize_range(0, 4),
            g.usize_range(0, 4),
            g.usize_range(0, 4),
        ];
        let v = g.i32_range(-100, 99);
        let a = Ndarr::<i32, U3>::fill(v, shape);
        assert_eq!(a.data().len(), product(&shape), "rand_fill[{trial}]");
        assert!(a.data().iter().all(|x| *x == v), "rand_fill[{trial}]");
        let a = Ndarr::<i32, U3>::zeros(shape);
        assert_eq!(a.data().len(), product(&shape), "rand_zeros[{trial}]");
        assert!(a.data().iter().all(|x| *x == 0), "rand_zeros[{trial}]");
    }
}

fn ref_linspace_f64(start: f64, end: f64, n: u16) -> Vec<f64> {
    if n == 1 {
        return vec![start];
    }
    let dx = (end - start) / f64::from(n - 1);
    (0..n).map(|i| start + f64::from(i) * dx).collect()
}

#[test]
fn linspace_f64() {
    for (start, end, n) in [
        (0.0, 9.0, 10u16),
        (-5.0, 5.0, 11),
        (5.0, -5.0, 7),
        (0.0, 1.0, 2),
        (1.0, 1.0, 4),
        (-1.5, 2.5, 3),
        (0.0, 1.0, 1),
        (1e-8, 1e8, 5),
    ] {
        let a = Ndarr::<f64, U1>::linspace(start, end, n);
        assert_eq!(a.shape(), &[n as usize]);
        assert_eq!(a.data().len(), n as usize);
        assert_close(
            &format!("linspace_f64({start},{end},{n})"),
            a.data(),
            &ref_linspace_f64(start, end, n),
            1e-9,
        );
        assert_eq!(a.data()[0], start, "linspace starts exactly at start");
    }
}

#[test]
fn linspace_f32_and_i32() {
    let a = Ndarr::<f32, U1>::linspace(-2.0, 2.0, 9);
    let want: Vec<f32> = (0..9).map(|i| -2.0 + i as f32 * 0.5).collect();
    assert_close("linspace_f32", a.data(), &want, 1e-6);

    for (start, end, n) in [(0i32, 9i32, 10u16), (-6, 6, 7), (3, 3, 4)] {
        let a = Ndarr::<i32, U1>::linspace(start, end, n);
        // Integer linspace uses truncating division for the step.
        let dx = (end - start) / i32::from(n - 1);
        let want: Vec<i32> = (0..n).map(|i| start + i32::from(i) * dx).collect();
        assert_eq_slices(&format!("linspace_i32({start},{end},{n})"), a.data(), &want);
    }
}

#[test]
fn linspace_endpoint_drift() {
    // linspace accumulates `dx`, so the last element is not bit-exactly `end` but must stay close.
    let a = Ndarr::<f64, U1>::linspace(0.0, 1.0, 1001);
    assert_close(
        "linspace_drift",
        a.data(),
        &ref_linspace_f64(0.0, 1.0, 1001),
        1e-9,
    );
    let last = *a.data().last().unwrap();
    assert!(
        (last - 1.0).abs() < 1e-12,
        "linspace endpoint drifted too far: {last}"
    );
}

// `n - 1` underflows u16 at n = 0; debug builds panic on the check.
#[test]
#[should_panic(expected = "attempt to subtract with overflow")]
fn linspace_degenerate_n0() {
    let _ = Ndarr::<f64, U1>::linspace(0.0, 1.0, 0);
}

// integer T divides by zero at n = 1.
#[test]
#[should_panic(expected = "attempt to divide by zero")]
fn linspace_degenerate_i32_n1() {
    let _ = Ndarr::<i32, U1>::linspace(0, 9, 1);
}

#[test]
fn logspace_values() {
    for (start, end, base, n) in [
        (0.0, 9.0, 10.0, 10u16),
        (-3.0, 3.0, 2.0, 7),
        (0.0, 1.0, 5.0, 2),
        (1.0, 1.0, 3.0, 5),
        (2.0, -2.0, 10.0, 5),
    ] {
        let a = logspace::<f64>(start, end, base, n);
        assert_eq!(a.shape(), &[n as usize]);
        let step = (end - start) / f64::from(n - 1);
        let want: Vec<f64> = (0..n)
            .map(|i| base.powf(start + step * f64::from(i)))
            .collect();
        assert_close(
            &format!("logspace({start},{end},{base},{n})"),
            a.data(),
            &want,
            1e-12,
        );
    }
    let a = logspace::<f32>(0.0f32, 4.0, 10.0, 5);
    let want: Vec<f32> = (0..5).map(|i| 10.0f32.powf(i as f32)).collect();
    assert_close("logspace_f32", a.data(), &want, 1e-5);
}

fn ref_geomspace_f64(start: f64, end: f64, n: u16) -> Vec<f64> {
    let segments = f64::from(n - 1);
    let ratio = (end / start).powf(1.0 / segments);
    (0..n).map(|i| start * ratio.powf(f64::from(i))).collect()
}

#[test]
fn geomspace_values() {
    for (start, end, n) in [
        (1.0, 256.0, 9u16),
        (1.0, 1000.0, 4),
        (2.0, 32.0, 5),
        (-1.0, -64.0, 7),
        (1.0, 1.0, 3),
    ] {
        let a = geomspace::<f64>(start, end, n);
        assert_eq!(a.shape(), &[n as usize]);
        assert_close(
            &format!("geomspace({start},{end},{n})"),
            a.data(),
            &ref_geomspace_f64(start, end, n),
            1e-12,
        );
    }
    let a = geomspace::<f32>(1.0f32, 64.0, 7);
    let want: Vec<f32> = (0..7).map(|i| 2.0f32.powf(i as f32)).collect();
    assert_close("geomspace_f32", a.data(), &want, 1e-5);
}

#[test]
fn geomspace_degenerate() {
    // n == 1 makes `segments` zero; ratio.powf(0) == 1, so the element is `start`.
    let a = geomspace::<f64>(1.0, 8.0, 1);
    assert_eq!(a.data(), &[1.0]);

    // start == 0 makes the ratio infinite; every element after the first is NaN.
    let b = geomspace::<f64>(0.0, 8.0, 4);
    assert_eq!(b.data()[0], 0.0);
    assert!(
        b.data()[1..].iter().all(|v| v.is_nan()),
        "geomspace(0., ..) tail must be NaN, got {:?}",
        b.data()
    );
}

#[test]
fn spaces_random() {
    let mut g = Lcg::new(0x5AC3_5EED);
    for trial in 0..24 {
        let n = g.usize_range(2, 32) as u16;
        let start = g.f64_range(-10.0, 10.0);
        let end = g.f64_range(-10.0, 10.0);
        let a = Ndarr::<f64, U1>::linspace(start, end, n);
        assert_close(
            &format!("rand_linspace[{trial}]"),
            a.data(),
            &ref_linspace_f64(start, end, n),
            1e-9,
        );

        let base = g.f64_range(1.5, 8.0);
        let a = logspace::<f64>(start, end, base, n);
        let step = (end - start) / f64::from(n - 1);
        let want: Vec<f64> = (0..n)
            .map(|i| base.powf(start + step * f64::from(i)))
            .collect();
        assert_close(&format!("rand_logspace[{trial}]"), a.data(), &want, 1e-9);

        let gs = g.f64_range(0.5, 4.0);
        let ge = g.f64_range(8.0, 64.0);
        let a = geomspace::<f64>(gs, ge, n);
        assert_close(
            &format!("rand_geomspace[{trial}]"),
            a.data(),
            &ref_geomspace_f64(gs, ge, n),
            1e-9,
        );
    }
}

/// Global softmax built from primitives over one flattened lane. Panics on empty input.
fn softmax<T: Float, R: Rank>(a: &Ndarr<T, R>) -> Ndarr<T, R> {
    let flat = a.view().reshape([a.len()]).unwrap();
    let maxima = flat.reduce(0, T::max).unwrap();
    let maxima = maxima.insert_axis_view(0).unwrap();
    let exp = (&flat - &maxima).exp();
    let totals = exp.reduce(0, |acc, value| acc + value).unwrap();
    let totals = totals.insert_axis_view(0).unwrap();
    (&exp / &totals).reshape(a.dim().clone()).unwrap()
}

const ACT_GRID: [f64; 17] = [
    -1e3, -6.0, -3.5, -3.0, -1.5, -1.0, -0.5, -1e-9, 0.0, 1e-9, 0.5, 1.0, 1.5, 3.0, 3.5, 6.0, 1e3,
];

#[test]
fn softmax_grid_f64_and_f32() {
    let a = Ndarr::new(&ACT_GRID, [ACT_GRID.len()]).unwrap();
    assert_close(
        "softmax",
        softmax(&a).data(),
        &ref_softmax(&ACT_GRID),
        1e-12,
    );
    let sum: f64 = softmax(&a).data().iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-12,
        "softmax must sum to 1, got {sum}"
    );

    let data: Vec<f32> = ACT_GRID.iter().map(|x| *x as f32).collect();
    let a = Ndarr::<f32, U1>::new(&data, [data.len()]).unwrap();
    assert_close("softmax f32", softmax(&a).data(), &ref_softmax(&data), 1e-6);
}

#[test]
fn softmax_asymmetric_ranks() {
    let mut g = Lcg::new(0xAC71_0001);
    for shape in SHAPES_2_UTILS {
        let data = make_f64_lcg(product(&shape), &mut g);
        let a = Ndarr::new(&data, shape).unwrap();
        if product(&shape) > 0 {
            assert_eq!(softmax(&a).shape(), &shape);
            assert_close("softmax r2", softmax(&a).data(), &ref_softmax(&data), 1e-12);
        }
    }
    for shape in SHAPES_3_UTILS {
        let data = make_f64_lcg(product(&shape), &mut g);
        let a = Ndarr::new(&data, shape).unwrap();
        if product(&shape) > 0 {
            assert_eq!(softmax(&a).shape(), &shape);
            assert_close("softmax r3", softmax(&a).data(), &ref_softmax(&data), 1e-12);
        }
    }
}

#[test]
fn softmax_rank0_and_single_element() {
    let dim = Dim::<U0>::new(&[]).unwrap();
    let a = Ndarr::<f64, U0>::new(&[-0.75], &dim).unwrap();
    assert_eq!(softmax(&a).data(), &[1.0], "softmax of a single element");

    let b = Ndarr::new(&[3.25], [1]).unwrap();
    assert_eq!(softmax(&b).data(), &[1.0]);
}

#[test]
fn softmax_empty_panics() {
    // the seedless max reduce has no value on an empty buffer, so it panics.
    let a = Ndarr::<f64, U2>::zeros([0, 3]);
    assert!(panics(move || {
        let _ = softmax(&a);
    }));
}

#[test]
fn softmax_on_view_derived_inputs() {
    let mut g = Lcg::new(0xAC71_0002);
    let shape = [2usize, 3, 4];
    let data = make_f64_lcg(product(&shape), &mut g);
    let a = Ndarr::new(&data, shape).unwrap();

    // Transposed input matches a reference computed on independently transposed flat data.
    let ta = a.t_view().to_owned_array();
    let (tshape, tdata) = ref_transpose(&data, &shape);
    assert_eq!(ta.shape(), tshape.as_slice());
    assert_close(
        "softmax_after_t",
        softmax(&ta).data(),
        &ref_softmax(&tdata),
        1e-12,
    );

    for axis in 0..3 {
        for i in 0..shape[axis] {
            let va = a.index_axis_view(axis, i).unwrap().to_owned_array();
            let slice = ref_slice_at(&data, &shape, axis, i);
            assert_close(
                &format!("softmax_axis_view({axis},{i})"),
                softmax(&va).data(),
                &ref_softmax(&slice),
                1e-12,
            );
        }
    }

    let row = make_f64_lcg(4, &mut g);
    let r = Ndarr::new(&row, [4]).unwrap();
    let dim = Dim::<U3>::new(&shape).unwrap();
    let ba = r.broadcast_view_to(&dim).unwrap().to_owned_array();
    let bdata = ref_broadcast(&row, &[4], &shape);
    assert_close(
        "softmax_after_broadcast",
        softmax(&ba).data(),
        &ref_softmax(&bdata),
        1e-12,
    );
}

#[test]
fn softmax_random() {
    let mut g = Lcg::new(0xAC71_0003);
    for trial in 0..16 {
        let rows = g.usize_range(1, 5);
        let cols = g.usize_range(1, 5);
        let data = make_f64_lcg(rows * cols, &mut g);
        let a = Ndarr::new(&data, [rows, cols]).unwrap();
        assert_close(
            &format!("rand_softmax[{trial}]"),
            softmax(&a).data(),
            &ref_softmax(&data),
            1e-12,
        );
        let sum: f64 = softmax(&a).data().iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-12,
            "softmax must sum to 1, got {sum} for {rows}x{cols}"
        );
    }
}

#[cfg(feature = "fft")]
mod fft {
    use super::*;
    use rapl::C;

    fn to_c(v: &[(f64, f64)]) -> Vec<C<f64>> {
        v.iter().map(|&(re, im)| C(re, im)).collect()
    }

    fn c1(v: &[(f64, f64)]) -> Ndarr<C<f64>, U1> {
        Ndarr::new(&to_c(v), [v.len()]).unwrap()
    }

    fn c2(v: &[(f64, f64)], shape: [usize; 2]) -> Ndarr<C<f64>, U2> {
        Ndarr::new(&to_c(v), shape).unwrap()
    }

    fn make_c64(n: usize, g: &mut Lcg) -> Vec<(f64, f64)> {
        (0..n)
            .map(|_| (g.f64_range(-4.0, 4.0), g.f64_range(-4.0, 4.0)))
            .collect()
    }

    fn assert_close_pairs<T: Float + Debug>(op: &str, got: &[C<T>], want: &[(T, T)], eps: T) {
        assert_eq!(got.len(), want.len(), "{op}: length mismatch");
        for (i, (a, b)) in got.iter().zip(want.iter()).enumerate() {
            let ok = (a.0 - b.0).abs() <= eps && (a.1 - b.1).abs() <= eps;
            if !ok {
                panic!(
                    "{op}: mismatch vs reference at {i}: got=({:?},{:?}) want=({:?},{:?})",
                    a.0, a.1, b.0, b.1
                );
            }
        }
    }

    // Reference DFT independent of the crate: O(n^2) over plain arithmetic.
    fn naive_dft(x: &[(f64, f64)], inverse: bool) -> Vec<(f64, f64)> {
        let n = x.len();
        let sign = if inverse { 1.0 } else { -1.0 };
        (0..n)
            .map(|k| {
                let mut re = 0.0;
                let mut im = 0.0;
                for (j, (xr, xi)) in x.iter().enumerate() {
                    let ang =
                        sign * 2.0 * std::f64::consts::PI * (k as f64) * (j as f64) / (n as f64);
                    let (s, c) = ang.sin_cos();
                    re += xr * c - xi * s;
                    im += xr * s + xi * c;
                }
                (re, im)
            })
            .collect()
    }

    fn naive_dft2(x: &[(f64, f64)], rows: usize, cols: usize, inverse: bool) -> Vec<(f64, f64)> {
        let mut out = x.to_vec();
        for c in 0..cols {
            let col: Vec<(f64, f64)> = (0..rows).map(|r| out[r * cols + c]).collect();
            let t = naive_dft(&col, inverse);
            for r in 0..rows {
                out[r * cols + c] = t[r];
            }
        }
        for r in 0..rows {
            let row: Vec<(f64, f64)> = (0..cols).map(|c| out[r * cols + c]).collect();
            let t = naive_dft(&row, inverse);
            out[r * cols..(r + 1) * cols].copy_from_slice(&t);
        }
        out
    }

    #[test]
    fn fft_1d_lengths() {
        let mut g = Lcg::new(0xFF71_0001);
        // powers of two, primes, and composites with large prime factors
        for n in [
            1usize, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 16, 17, 24, 32, 33,
        ] {
            let data = make_c64(n, &mut g);
            let a = c1(&data);
            let fa = a.fft();
            assert_eq!(fa.shape(), &[n], "fft({n}) shape");
            assert_close_pairs(
                &format!("fft_vs_naive({n})"),
                fa.data(),
                &naive_dft(&data, false),
                1e-8,
            );
            let ia = fa.ifft();
            assert_close_pairs(&format!("roundtrip({n})"), ia.data(), &data, 1e-9);
        }
    }

    #[test]
    fn fft_1d_f32() {
        let mut g = Lcg::new(0xFF71_0002);
        for n in [2usize, 3, 8, 15] {
            let data = make_c64(n, &mut g);
            let pairs: Vec<(f32, f32)> = data.iter().map(|&(r, i)| (r as f32, i as f32)).collect();
            let cs: Vec<C<f32>> = pairs.iter().map(|&(r, i)| C(r, i)).collect();
            let a = Ndarr::new(&cs, [n]).unwrap();
            // fft against a f64 reference DFT of the (rounded) f32 inputs
            let inputs64: Vec<(f64, f64)> = pairs
                .iter()
                .map(|&(r, i)| (f64::from(r), f64::from(i)))
                .collect();
            let want = naive_dft(&inputs64, false);
            let fa = a.fft();
            for (i, (z, w)) in fa.data().iter().zip(&want).enumerate() {
                assert!(
                    (f64::from(z.0) - w.0).abs() < 1e-3 && (f64::from(z.1) - w.1).abs() < 1e-3,
                    "fft_f32({n}) mismatch at {i}"
                );
            }
            let ra = a.fft().ifft();
            assert_close_pairs(&format!("ifft_f32({n})"), ra.data(), &pairs, 1e-4);
        }
    }

    #[test]
    fn fft_1d_single_and_constant() {
        let a = c1(&[(2.5f64, -1.25)]);
        assert_eq!(a.fft().data()[0].0, 2.5, "length-1 fft is the identity");
        assert_eq!(a.fft().data()[0].1, -1.25);
        assert_close_pairs("ifft_len1", a.ifft().data(), &[(2.5, -1.25)], 1e-12);

        let constant: Vec<(f64, f64)> = vec![(1.0, 0.0); 6];
        let a = c1(&constant);
        let fa = a.fft();
        assert!(
            (fa.data()[0].0 - 6.0).abs() < 1e-12,
            "DC bin of a constant signal must be n"
        );
        for z in &fa.data()[1..] {
            assert!(
                z.0.abs() < 1e-12 && z.1.abs() < 1e-12,
                "non-DC bins must vanish"
            );
        }

        let mut impulse = vec![(0.0, 0.0); 8];
        impulse[0] = (1.0, 0.0);
        let fa = c1(&impulse).fft();
        for z in fa.data() {
            assert!(
                (z.0 - 1.0).abs() < 1e-12 && z.1.abs() < 1e-12,
                "impulse spectrum must be flat"
            );
        }
    }

    #[test]
    fn fft_1d_empty() {
        // Zero-length transforms are accepted and return an empty array.
        let a = Ndarr::<C<f64>, U1>::new(&[], [0]).unwrap();
        let fa = a.fft();
        assert_eq!(fa.shape(), &[0]);
        assert!(fa.data().is_empty());
        let ia = a.ifft();
        assert!(ia.data().is_empty());
    }

    const FFT2_SHAPES: [[usize; 2]; 11] = [
        [1, 1],
        [1, 5],
        [5, 1],
        [2, 3],
        [3, 2],
        [4, 6],
        [6, 4],
        [3, 5],
        [5, 3],
        [8, 3],
        [7, 4],
    ];

    #[test]
    fn fft2d_asymmetric() {
        let mut g = Lcg::new(0xFF71_0003);
        for shape in FFT2_SHAPES {
            let n = product(&shape);
            let data = make_c64(n, &mut g);
            let a = c2(&data, shape);
            let fa = a.fft2d();
            assert_eq!(fa.shape(), &shape, "fft2d {shape:?} shape");
            assert_close_pairs(
                &format!("fft2d_vs_naive({shape:?})"),
                fa.data(),
                &naive_dft2(&data, shape[0], shape[1], false),
                1e-8,
            );
            let ia = fa.ifft2();
            assert_close_pairs(
                &format!("fft2d_roundtrip({shape:?})"),
                ia.data(),
                &data,
                1e-9,
            );
        }
    }

    #[test]
    fn ifft2_forward_of_reference() {
        let mut g = Lcg::new(0xFF71_0004);
        for shape in [[2usize, 5], [5, 2], [3, 4]] {
            let n = product(&shape);
            let data = make_c64(n, &mut g);
            let a = c2(&data, shape);
            let ia = a.ifft2();
            let reference: Vec<(f64, f64)> = naive_dft2(&data, shape[0], shape[1], true)
                .into_iter()
                .map(|(re, im)| (re / n as f64, im / n as f64))
                .collect();
            assert_close_pairs(
                &format!("ifft2_vs_naive({shape:?})"),
                ia.data(),
                &reference,
                1e-8,
            );
        }
    }

    #[test]
    fn fft2d_f32() {
        let mut g = Lcg::new(0xFF71_0005);
        for shape in [[3usize, 5], [5, 3], [4, 4]] {
            let n = product(&shape);
            let pairs: Vec<(f32, f32)> = make_c64(n, &mut g)
                .into_iter()
                .map(|(r, i)| (r as f32, i as f32))
                .collect();
            let cs: Vec<C<f32>> = pairs.iter().map(|&(r, i)| C(r, i)).collect();
            let a = Ndarr::new(&cs, shape).unwrap();
            let ra = a.fft2d().ifft2();
            assert_close_pairs(
                &format!("fft2d_f32_roundtrip({shape:?})"),
                ra.data(),
                &pairs,
                1e-3,
            );
        }
    }

    #[test]
    fn fft2d_degenerate() {
        // Zero-axis shapes transform to themselves.
        for shape in [[0usize, 3], [3, 0], [0, 0]] {
            let a = c2(&[], shape);
            assert_eq!(a.fft2d().shape(), &shape);
            assert!(a.fft2d().data().is_empty());
            assert_eq!(a.ifft2().shape(), &shape);
            assert!(a.ifft2().data().is_empty());
        }
    }

    #[test]
    fn fft2d_from_view_derived_inputs() {
        let mut g = Lcg::new(0xFF71_0006);
        let shape = [3usize, 5];
        let data = make_c64(product(&shape), &mut g);
        let a = c2(&data, shape);

        // Transposed shape is [5, 3], so the row/column passes swap lengths.
        let ta = a.t_view().to_owned_array();
        assert_eq!(ta.shape(), &[5, 3]);
        let mut transposed: Vec<(f64, f64)> = Vec::with_capacity(15);
        for c in 0..5 {
            for r in 0..3 {
                transposed.push(data[r * 5 + c]);
            }
        }
        assert_close_pairs(
            "fft2d_after_t_vs_naive",
            ta.fft2d().data(),
            &naive_dft2(&transposed, 5, 3, false),
            1e-8,
        );
        let n = 15.0;
        let inverse_ref: Vec<(f64, f64)> = naive_dft2(&transposed, 5, 3, true)
            .into_iter()
            .map(|(re, im)| (re / n, im / n))
            .collect();
        assert_close_pairs(
            "ifft2_after_t_vs_naive",
            ta.ifft2().data(),
            &inverse_ref,
            1e-8,
        );

        // Rank-1 fft of an axis view (strided before materialization).
        for axis in 0..2 {
            for i in 0..shape[axis] {
                let va = a.index_axis_view(axis, i).unwrap().to_owned_array();
                let lane: Vec<(f64, f64)> = ref_slice_at(&data, &shape, axis, i);
                assert_close_pairs(
                    &format!("fft_axis_view({axis},{i})"),
                    va.fft().data(),
                    &naive_dft(&lane, false),
                    1e-8,
                );
            }
        }

        // Broadcast a row into [3, 5] (0-stride axis before materialization).
        let row = make_c64(5, &mut g);
        let r = c1(&row);
        let dim = Dim::<U2>::new(&shape).unwrap();
        let ba = r.broadcast_view_to(&dim).unwrap().to_owned_array();
        let bdata = ref_broadcast(&row, &[5], &shape);
        assert_close_pairs(
            "fft2d_after_broadcast",
            ba.fft2d().data(),
            &naive_dft2(&bdata, shape[0], shape[1], false),
            1e-8,
        );
    }

    #[test]
    fn fft2d_large_prime_shapes() {
        let mut g = Lcg::new(0xFF71_0007);
        for shape in [[11usize, 13], [13, 11], [16, 9], [9, 16], [17, 2], [2, 17]] {
            let n = product(&shape);
            let data = make_c64(n, &mut g);
            let a = c2(&data, shape);
            let fa = a.fft2d();
            assert_close_pairs(
                &format!("fft2d_prime_vs_naive({shape:?})"),
                fa.data(),
                &naive_dft2(&data, shape[0], shape[1], false),
                1e-7,
            );
            assert_close_pairs(
                &format!("fft2d_prime_roundtrip({shape:?})"),
                fa.ifft2().data(),
                &data,
                1e-9,
            );
        }
    }

    #[test]
    fn fft2d_equals_composed_1d_passes() {
        // fft2d must equal a column pass then a row pass of the crate's own rank-1 fft.
        let mut g = Lcg::new(0xFF71_0008);
        for shape in [[3usize, 5], [5, 3], [1, 6], [6, 1], [4, 7]] {
            let (rows, cols) = (shape[0], shape[1]);
            let data = make_c64(rows * cols, &mut g);
            let a = c2(&data, shape);
            let mut work = data.clone();
            for c in 0..cols {
                let col: Vec<(f64, f64)> = (0..rows).map(|r| work[r * cols + c]).collect();
                let t = c1(&col).fft();
                for r in 0..rows {
                    work[r * cols + c] = (t.data()[r].0, t.data()[r].1);
                }
            }
            for r in 0..rows {
                let row: Vec<(f64, f64)> = (0..cols).map(|c| work[r * cols + c]).collect();
                let t = c1(&row).fft();
                for c in 0..cols {
                    work[r * cols + c] = (t.data()[c].0, t.data()[c].1);
                }
            }
            assert_close_pairs(
                &format!("fft2d_vs_composed({shape:?})"),
                a.fft2d().data(),
                &work,
                1e-9,
            );
        }
    }

    #[test]
    fn fft_random_differential() {
        let mut g = Lcg::new(0xFF71_0009);
        for trial in 0..16 {
            let n = g.usize_range(1, 20);
            let data = make_c64(n, &mut g);
            let a = c1(&data);
            assert_close_pairs(
                &format!("rand_fft[{trial}] n={n}"),
                a.fft().data(),
                &naive_dft(&data, false),
                1e-8,
            );
            let inverse_ref: Vec<(f64, f64)> = naive_dft(&data, true)
                .into_iter()
                .map(|(re, im)| (re / n as f64, im / n as f64))
                .collect();
            assert_close_pairs(
                &format!("rand_ifft[{trial}] n={n}"),
                a.ifft().data(),
                &inverse_ref,
                1e-8,
            );
            assert_close_pairs(
                &format!("rand_roundtrip[{trial}] n={n}"),
                a.fft().ifft().data(),
                &data,
                1e-9,
            );

            let rows = g.usize_range(1, 6);
            let cols = g.usize_range(1, 6);
            let data2 = make_c64(rows * cols, &mut g);
            let a2 = c2(&data2, [rows, cols]);
            assert_close_pairs(
                &format!("rand_fft2d_naive[{trial}] {rows}x{cols}"),
                a2.fft2d().data(),
                &naive_dft2(&data2, rows, cols, false),
                1e-7,
            );
            assert_close_pairs(
                &format!("rand_fft2d_roundtrip[{trial}] {rows}x{cols}"),
                a2.fft2d().ifft2().data(),
                &data2,
                1e-9,
            );
        }
    }

    fn ref_fftshift1<T: Clone>(data: &[T]) -> Vec<T> {
        let n = data.len();
        (0..n)
            .map(|i| data[(i + n.div_ceil(2)) % n].clone())
            .collect()
    }

    fn ref_fftshift2<T: Clone>(data: &[T], m: usize, k: usize) -> Vec<T> {
        (0..data.len())
            .map(|i| {
                let (row, col) = (i / k, i % k);
                let sr = (row + m.div_ceil(2)) % m;
                let sc = (col + k.div_ceil(2)) % k;
                data[sr * k + sc].clone()
            })
            .collect()
    }

    #[test]
    fn fftshift_1d() {
        for n in 1usize..=10 {
            let data: Vec<i32> = (0..n as i32).collect();
            let a = Ndarr::new(&data, [n]).unwrap();
            let sa = a.fftshift();
            assert_eq!(sa.shape(), &[n], "fftshift({n}) shape");
            // rotate-left by ceil(n/2), matching numpy's fftshift
            assert_eq_slices(
                &format!("fftshift_1d({n})"),
                sa.data(),
                &ref_fftshift1(&data),
            );
        }
    }

    #[test]
    fn fftshift_1d_empty() {
        let a = Ndarr::<i32, U1>::zeros([0]);
        let sa = a.fftshift();
        assert_eq!(sa.shape(), &[0]);
        assert!(sa.data().is_empty());
    }

    #[test]
    fn fftshift_2d_asymmetric() {
        for shape in [
            [1usize, 1],
            [1, 5],
            [5, 1],
            [2, 2],
            [3, 4],
            [4, 3],
            [5, 3],
            [3, 5],
            [6, 7],
        ] {
            let n = product(&shape);
            let data: Vec<i32> = (0..n as i32).collect();
            let a = Ndarr::new(&data, shape).unwrap();
            let sa = a.fftshift();
            assert_eq!(sa.shape(), &shape, "fftshift {shape:?} shape");
            assert_eq_slices(
                &format!("fftshift_2d({shape:?})"),
                sa.data(),
                &ref_fftshift2(&data, shape[0], shape[1]),
            );
        }
    }

    #[test]
    fn fftshift_2d_degenerate() {
        // Zero-axis inputs shift to (empty) outputs of the same shape.
        for shape in [[0usize, 3], [3, 0]] {
            let a = Ndarr::<i32, U2>::zeros(shape);
            let sa = a.fftshift();
            assert_eq!(sa.shape(), &shape);
            assert!(sa.data().is_empty());
        }
    }

    #[test]
    fn fftshift_after_transpose() {
        let shape = [3usize, 5];
        let data: Vec<i32> = (0..15).collect();
        let a = Ndarr::new(&data, shape).unwrap();
        let ta = a.t_view().to_owned_array();
        let (_, tdata) = ref_transpose(&data, &shape);
        assert_eq_slices(
            "fftshift_after_t",
            ta.fftshift().data(),
            &ref_fftshift2(&tdata, 5, 3),
        );
        assert_eq!(ta.fftshift(), a.t().fftshift());
        for i in 0..3 {
            let va = a.index_axis_view(0, i).unwrap().to_owned_array();
            let lane = ref_slice_at(&data, &shape, 0, i);
            assert_eq_slices(
                &format!("fftshift_axis_view({i})"),
                va.fftshift().data(),
                &ref_fftshift1(&lane),
            );
        }
    }

    #[test]
    fn fftshift_random() {
        let mut g = Lcg::new(0xFF71_000A);
        for trial in 0..24 {
            let n = g.usize_range(1, 17);
            let data: Vec<i32> = (0..n as i32).map(|i| i * 3 - 5).collect();
            let a = Ndarr::new(&data, [n]).unwrap();
            assert_eq_slices(
                &format!("rand_fftshift_1d[{trial}]"),
                a.fftshift().data(),
                &ref_fftshift1(&data),
            );

            let rows = g.usize_range(1, 6);
            let cols = g.usize_range(1, 6);
            let data2: Vec<i32> = (0..(rows * cols) as i32).collect();
            let a2 = Ndarr::new(&data2, [rows, cols]).unwrap();
            assert_eq_slices(
                &format!("rand_fftshift_2d[{trial}]"),
                a2.fftshift().data(),
                &ref_fftshift2(&data2, rows, cols),
            );
        }
    }
}

#[test]
fn new_element_count_validation() {
    for shape in SHAPES_3_API {
        let n = product(&shape);
        let data = make_i32(&shape, 3);
        assert!(Ndarr::new(&data, shape).is_ok());

        let short = &data[..n - 1];
        assert!(
            Ndarr::new(short, shape).is_err(),
            "new must reject {} elements for {shape:?}",
            n - 1
        );

        let mut long = data.clone();
        long.push(0);
        assert!(Ndarr::new(&long, shape).is_err());
    }
}

#[test]
fn new_zero_axis_shapes() {
    for shape in [[0usize, 3], [3, 0], [0, 0], [0, 1]] {
        let a = Ndarr::new(&[] as &[i32], shape).unwrap();
        assert_eq!(a.shape(), &shape);
        assert_eq!(a.len(), 0);
        assert!(a.data().is_empty());
        assert!(Ndarr::new(&[1i32], shape).is_err());
    }
}

#[test]
fn from_nested_and_range() {
    let a = Ndarr::from([[1, 2, 3], [4, 5, 6]]);
    assert_eq!(a.shape(), &[2, 3]);
    assert_eq_slices("from_nested", a.data(), &[1, 2, 3, 4, 5, 6]);

    let ra = Ndarr::from(0..7);
    assert_eq_slices("from_range", ra.data(), &(0..7).collect::<Vec<i32>>());

    let sa = Ndarr::from([[[1, 2]], [[3, 4]], [[5, 6]]]);
    assert_eq!(sa.shape(), &[3, 1, 2]);
    assert_eq_slices("from_nested3", sa.data(), &[1, 2, 3, 4, 5, 6]);
}

#[test]
fn reshape_asymmetric() {
    let shape = [2usize, 3, 4];
    let data = make_i32(&shape, 1);
    let a = Ndarr::new(&data, shape).unwrap();

    // Reshaping a contiguous array keeps the flat buffer for every target shape.
    for target in [[4usize, 6], [24, 1], [1, 24], [6, 4], [12, 2], [8, 3]] {
        let ra = a.clone().reshape(target).unwrap();
        assert_eq!(ra.shape(), target);
        assert_eq_slices("reshape_u2", ra.data(), &data);
    }

    for target in [[3usize, 2, 4], [4, 3, 2], [1, 24, 1], [24, 1, 1]] {
        let ra = a.clone().reshape(target).unwrap();
        assert_eq!(ra.shape(), target);
        assert_eq_slices("reshape_u3", ra.data(), &data);
    }

    let flat = a.reshape([24]).unwrap();
    assert_eq_slices("reshape_flat_is_data", flat.data(), &data);
}

#[test]
fn reshape_error_paths() {
    let a = Ndarr::new(&make_i32(&[2, 3], 0), [2, 3]).unwrap();
    for bad in [[2usize, 2], [7, 1], [6, 2], [1, 5]] {
        assert!(a.view().reshape(bad).is_err(), "reshape [2,3] -> {bad:?}");
    }
    assert!(a.view().reshape([5]).is_err());
    assert!(a.view().reshape([6]).is_ok());
}

#[test]
fn reshape_between_zero_axis_shapes() {
    let a = Ndarr::new(&[] as &[i32], [0, 3]).unwrap();
    // reshape only compares element counts, so 0-element shapes interconvert.
    let ra = a.clone().reshape([5, 0]).unwrap();
    assert_eq!(ra.shape(), &[5, 0]);
    assert!(ra.data().is_empty());
    assert!(a.reshape([3]).is_err());
}

#[test]
fn borrowed_and_owned_reshape_agree() {
    for shape in SHAPES_3_API {
        let n = product(&shape);
        let data = make_i32(&shape, 5);
        let a = Ndarr::new(&data, shape).unwrap();
        let target = [n, 1];
        let view = a.view().reshape(target).unwrap();
        let eager = a.clone().reshape(target).unwrap();
        assert_eq!(view.shape(), &target);
        assert_eq_slices(
            "reshape_view_vs_eager",
            view.to_owned_array().data(),
            eager.data(),
        );
        assert_eq_slices("reshape_view_is_data", view.to_owned_array().data(), &data);
        for i in 0..n {
            assert_eq!(
                view[[i, 0]],
                eager[[i, 0]],
                "reshape_view element {i} for {shape:?}"
            );
        }
    }
}

#[test]
fn reshape_view_error_paths() {
    let a = Ndarr::new(&make_i32(&[3, 4], 0), [3, 4]).unwrap();
    assert!(a.view().reshape([12]).is_ok());
    assert!(a.view().reshape([2, 6]).is_ok());
    assert!(a.view().reshape([5, 2]).is_err());
    assert!(a.view().reshape([13]).is_err());

    let empty = Ndarr::new(&[] as &[i32], [0, 4]).unwrap();
    let v = empty.view().reshape([4, 0]).unwrap();
    assert_eq!(v.shape(), &[4, 0]);
    assert_eq!(v.iter_elems().count(), 0);
}

#[test]
fn slice_at_asymmetric_all_axes() {
    for shape in SHAPES_3_API {
        let data = make_i32(&shape, 2);
        let a = Ndarr::new(&data, shape).unwrap();
        for axis in 0..3 {
            let sa = a.slice_at(axis);
            assert_eq!(sa.len(), shape[axis], "slice count {shape:?} axis {axis}");
            let mut expected_shape = shape.to_vec();
            expected_shape.remove(axis);
            for (i, x) in sa.iter().enumerate() {
                assert_eq!(x.shape(), expected_shape.as_slice());
                assert_eq_slices(
                    &format!("slice_at({axis})[{i}] {shape:?}"),
                    x.data(),
                    &ref_slice_at(&data, &shape, axis, i),
                );
            }
        }
    }
}

#[test]
fn slice_at_rank1_gives_rank0() {
    let data = make_i32(&[5], 11);
    let a = Ndarr::new(&data, [5]).unwrap();
    let sa = a.slice_at(0);
    assert_eq!(sa.len(), 5);
    for (i, x) in sa.iter().enumerate() {
        assert!(x.shape().is_empty());
        assert_eq!(x.clone().scalar(), data[i], "rank0 slice {i}");
    }
}

#[test]
fn slice_at_axis_out_of_bounds_panics() {
    let a = Ndarr::new(&make_i32(&[2, 3], 0), [2, 3]).unwrap();
    let b = a.clone();
    assert!(panics(move || {
        let _ = a.slice_at(2);
    }));
    assert!(panics(move || {
        let _ = b.slice_at(9);
    }));
}

#[test]
fn slice_at_length_one_axes() {
    for shape in [[1usize, 5], [5, 1], [1, 1]] {
        let data = make_i32(&shape, 4);
        let a = Ndarr::new(&data, shape).unwrap();
        for axis in 0..2 {
            let sa = a.slice_at(axis);
            assert_eq!(sa.len(), shape[axis]);
            for (i, x) in sa.iter().enumerate() {
                assert_eq_slices(
                    &format!("len1_slice({axis})[{i}] {shape:?}"),
                    x.data(),
                    &ref_slice_at(&data, &shape, axis, i),
                );
            }
        }
    }
}

#[test]
fn reduce_asymmetric_all_axes() {
    for shape in SHAPES_3_API {
        let data = make_i32(&shape, 1);
        let a = Ndarr::new(&data, shape).unwrap();
        for axis in 0..3 {
            let ra = a.reduce(axis, |x, y| x + y).unwrap();
            let (want_shape, want) = ref_reduce(&data, &shape, axis, |x, y| x + y);
            assert_eq!(ra.shape(), want_shape.as_slice());
            assert_eq_slices(&format!("reduce_add({axis}) {shape:?}"), ra.data(), &want);

            // Non-commutative reducer: order of folding is part of the contract.
            let la = a.reduce(axis, |x, y| x - 2 * y).unwrap();
            let (_, want) = ref_reduce(&data, &shape, axis, |x, y| x - 2 * y);
            assert_eq_slices(
                &format!("reduce_noncommutative({axis}) {shape:?}"),
                la.data(),
                &want,
            );
        }
    }
}

#[test]
fn reduce_rank1_and_float() {
    let data = make_i32(&[7], 3);
    let a = Ndarr::new(&data, [7]).unwrap();
    let ra = a.reduce(0, |x, y| x + y).unwrap();
    assert!(ra.shape().is_empty());
    assert_eq!(ra.scalar(), data.iter().sum::<i32>());

    let mut g = Lcg::new(0x4ED0_0001);
    let shape = [3usize, 5];
    let data = make_f64_lcg(product(&shape), &mut g);
    let fa = Ndarr::new(&data, shape).unwrap();
    for axis in 0..2 {
        let ra = fa.reduce(axis, |x, y| x + y).unwrap();
        let (_, want) = ref_reduce(&data, &shape, axis, |x, y| x + y);
        assert_close(&format!("reduce_f64({axis})"), ra.data(), &want, 1e-12);
    }
}

#[test]
fn reduce_axis_error_paths() {
    let a = Ndarr::new(&make_i32(&[2, 3], 0), [2, 3]).unwrap();
    for axis in [2usize, 3, 17] {
        assert!(a.reduce(axis, |x, y| x + y).is_err(), "reduce axis {axis}");
    }
    assert!(a.reduce(1, |x, y| x + y).is_ok());
}

#[test]
fn reduce_over_zero_length_axis() {
    // The empty reduction is reported as an error rather than a panic.
    let a = Ndarr::new(&[] as &[i32], [2, 0]).unwrap();
    assert!(a.reduce(1, |x, y| x + y).is_err());
}

#[test]
fn broadcast_asymmetric() {
    let cases: [(&[usize], [usize; 3]); 7] = [
        (&[4], [2, 3, 4]),
        (&[1], [2, 3, 4]),
        (&[3, 4], [2, 3, 4]),
        (&[1, 4], [2, 3, 4]),
        (&[3, 1], [2, 3, 4]),
        (&[1, 1], [5, 1, 3]),
        (&[5, 1, 3], [5, 1, 3]),
    ];
    for (src, target) in cases {
        let data = make_i32(src, 9);
        let want = ref_broadcast(&data, src, &target);
        let tdim = Dim::<U3>::new(&target).unwrap();
        match src.len() {
            1 => {
                let a = Ndarr::new(&data, [src[0]]).unwrap();
                let ba = a.broadcast(target).unwrap();
                assert_eq!(ba.shape(), &target);
                assert_eq_slices("broadcast_u1", ba.data(), &want);
                assert_eq_slices(
                    "broadcast_view_to_u1",
                    a.broadcast_view_to(&tdim).unwrap().to_owned_array().data(),
                    &want,
                );
            }
            2 => {
                let a = Ndarr::new(&data, [src[0], src[1]]).unwrap();
                let ba = a.broadcast(target).unwrap();
                assert_eq!(ba.shape(), &target);
                assert_eq_slices("broadcast_u2", ba.data(), &want);
                assert_eq_slices(
                    "broadcast_view_to_u2",
                    a.broadcast_view_to(&tdim).unwrap().to_owned_array().data(),
                    &want,
                );
            }
            _ => {
                let a = Ndarr::new(&data, [src[0], src[1], src[2]]).unwrap();
                let ba = a.broadcast(target).unwrap();
                assert_eq_slices("broadcast_u3", ba.data(), &want);
            }
        }
    }
}

#[test]
fn broadcast_error_paths() {
    let a = Ndarr::new(&make_i32(&[3], 0), [3]).unwrap();
    for bad in [[4usize], [2]] {
        assert!(a.broadcast(bad).is_err(), "broadcast [3] -> {bad:?}");
    }
    let c = Ndarr::new(&make_i32(&[2, 3], 0), [2, 3]).unwrap();
    for bad in [[2usize, 4], [3, 3], [5, 3]] {
        assert!(c.broadcast(bad).is_err(), "broadcast [2,3] -> {bad:?}");
        assert!(c.broadcast_to(bad).is_err());
    }
    // Target of lower rank than the broadcast result is rejected by `broadcast_to` only.
    assert!(c.broadcast_to([3usize]).is_err());
    assert!(c.broadcast([3usize]).is_ok());
}

#[test]
fn broadcast_to_when_tiling_is_valid() {
    let data = make_i32(&[4], 6);
    let a = Ndarr::new(&data, [4]).unwrap();
    let ba = a.broadcast_to([2usize, 3, 4]).unwrap();
    assert_eq_slices(
        "broadcast_to_leading",
        ba.data(),
        &ref_broadcast(&data, &[4], &[2, 3, 4]),
    );

    let data = make_i32(&[1, 5], 2);
    let c = Ndarr::new(&data, [1, 5]).unwrap();
    let bc = c.broadcast_to([3usize, 5]).unwrap();
    assert_eq_slices(
        "broadcast_to_row",
        bc.data(),
        &ref_broadcast(&data, &[1, 5], &[3, 5]),
    );

    let data = make_i32(&[2, 3], 1);
    let e = Ndarr::new(&data, [2, 3]).unwrap();
    let be = e.broadcast_to([2usize, 3]).unwrap();
    assert_eq_slices("broadcast_to_identity", be.data(), &data);
}

#[test]
fn broadcast_to_inner_size_one_axis_matches_numpy() {
    // [3,1] -> [3,2] must repeat each row element, i.e. [a,a,b,b,c,c].
    let a = Ndarr::new(&[1, 2, 3], [3, 1]).unwrap();
    let out = a.broadcast_to([3usize, 2]).unwrap();
    assert_eq!(out.shape(), &[3, 2]);
    assert_eq_slices("broadcast_to_inner_axis", out.data(), &[1, 1, 2, 2, 3, 3]);
    assert_eq_slices(
        "broadcast_inner_axis",
        a.broadcast([3usize, 2]).unwrap().data(),
        &[1, 1, 2, 2, 3, 3],
    );
}

#[test]
fn broadcast_zero_axis() {
    let a = Ndarr::new(&[] as &[i32], [0, 3]).unwrap();
    let ba = a.broadcast([0usize, 3]).unwrap();
    assert_eq!(ba.shape(), &[0, 3]);
    assert!(ba.data().is_empty());
}

#[test]
fn broadcast_rank_padding() {
    // Mirrors how the generalized products broadcast to right-aligned padded shapes.
    let data = make_i32(&[3, 1], 2);
    let a = Ndarr::new(&data, [3, 1]).unwrap();
    let targets: [&[usize]; 3] = [&[3, 4], &[1, 3, 4], &[2, 3, 4]];
    for target in targets {
        let da = a.broadcast(Dim::<Dyn>::new(target).unwrap()).unwrap();
        assert_eq_slices(
            "broadcast_padded",
            da.data(),
            &ref_broadcast(&data, &[3, 1], target),
        );
    }
}

#[test]
fn transpose_asymmetric() {
    for shape in SHAPES_3_API {
        let data = make_i32(&shape, 8);
        let a = Ndarr::new(&data, shape).unwrap();
        let ta = a.t();
        let (want_shape, want) = ref_transpose(&data, &shape);
        assert_eq!(ta.shape(), want_shape.as_slice());
        assert_eq_slices(&format!("t {shape:?}"), ta.data(), &want);
        assert_eq_slices("t_twice", ta.t().data(), &data);
        let tv = a.t_view();
        assert_eq!(
            tv.iter_elems().cloned().collect::<Vec<_>>(),
            ta.iter_elems().cloned().collect::<Vec<_>>(),
            "t_view vs t {shape:?}"
        );
    }
    for shape in SHAPES_2_API {
        let data = make_i32(&shape, 3);
        let a = Ndarr::new(&data, shape).unwrap();
        let (_, want) = ref_transpose(&data, &shape);
        assert_eq_slices(&format!("t2 {shape:?}"), a.t().data(), &want);
    }
}

#[test]
fn transpose_rank0_and_rank1() {
    let a = rank0(42);
    let ta = a.t();
    assert!(ta.shape().is_empty());
    assert_eq!(ta.scalar(), 42);

    let data = make_i32(&[6], 1);
    let c = Ndarr::new(&data, [6]).unwrap();
    assert_eq_slices("t_rank1", c.t().data(), &data);
    let e = Ndarr::new(&[99], [1]).unwrap();
    assert_eq_slices("t_single", e.t().data(), &[99]);
}

#[test]
fn transpose_zero_axis_shape() {
    // `t()` walks strides and handles empty arrays.
    let a = Ndarr::new(&[] as &[i32], [0, 3]).unwrap();
    let ta = a.t();
    assert_eq!(ta.shape(), &[3, 0]);
    assert!(ta.data().is_empty());
}

#[test]
fn roll_exhaustive_shifts() {
    for shape in SHAPES_3_API {
        let data = make_i32(&shape, 4);
        let a = Ndarr::new(&data, shape).unwrap();
        for axis in 0..3 {
            for shift in -9isize..=9 {
                let ra = a.roll(shift, axis);
                assert_eq!(ra.shape(), &shape);
                assert_eq_slices(
                    &format!("roll({shift},{axis}) {shape:?}"),
                    ra.data(),
                    &ref_roll(&data, &shape, shift, axis),
                );
            }
            // Rolling by the axis length is the identity.
            assert_eq_slices(
                &format!("roll_period({axis}) {shape:?}"),
                a.roll(shape[axis] as isize, axis).data(),
                &data,
            );
        }
    }
}

#[test]
fn roll_rank1_and_len1_axes() {
    let data = make_i32(&[5], 2);
    let a = Ndarr::new(&data, [5]).unwrap();
    for shift in [-6isize, -1, 0, 1, 4, 5, 11] {
        assert_eq_slices(
            &format!("roll1({shift})"),
            a.roll(shift, 0).data(),
            &ref_roll(&data, &[5], shift, 0),
        );
    }
    for shape in [[1usize, 5], [5, 1], [1, 1]] {
        let d = make_i32(&shape, 7);
        let c = Ndarr::new(&d, shape).unwrap();
        for axis in 0..2 {
            for shift in [-3isize, -1, 0, 2, 7] {
                assert_eq_slices(
                    &format!("roll_len1({shift},{axis}) {shape:?}"),
                    c.roll(shift, axis).data(),
                    &ref_roll(&d, &shape, shift, axis),
                );
            }
        }
    }
}

#[test]
fn roll_empty_axis_is_stable() {
    let a = Ndarr::new(&[] as &[i32], [2, 0]).unwrap();
    assert_eq!(a.roll(1, 1).shape(), &[2, 0]);

    let c = Ndarr::new(&make_i32(&[2, 3], 0), [2, 3]).unwrap();
    assert!(panics(move || {
        let _ = c.roll(1, 5);
    }));
}

#[test]
fn stack_all_source_and_insert_axes() {
    for shape in SHAPES_2_API {
        let data = make_i32(&shape, 5);
        let a = Ndarr::new(&data, shape).unwrap();
        for src_axis in 0..2 {
            let sa = a.slice_at(src_axis);
            let mut slice_shape = shape.to_vec();
            slice_shape.remove(src_axis);
            let slices_ref: Vec<Vec<i32>> = (0..shape[src_axis])
                .map(|i| ref_slice_at(&data, &shape, src_axis, i))
                .collect();
            for insert_axis in 0..2 {
                let da = stack(&sa, insert_axis);
                let (want_shape, want) = ref_stack(&slices_ref, &slice_shape, insert_axis);
                assert_eq!(da.shape(), want_shape.as_slice());
                assert_eq_slices(
                    &format!("stack({src_axis}->{insert_axis}) {shape:?}"),
                    da.data(),
                    &want,
                );
            }
            let round = stack(&sa, src_axis);
            assert_eq_slices(
                &format!("stack_roundtrip({src_axis}) {shape:?}"),
                round.data(),
                &data,
            );
        }
    }
}

#[test]
fn stack_rank3_sources() {
    for shape in SHAPES_3_API {
        let data = make_i32(&shape, 6);
        let a = Ndarr::new(&data, shape).unwrap();
        for src_axis in 0..3 {
            let sa = a.slice_at(src_axis);
            let mut slice_shape = shape.to_vec();
            slice_shape.remove(src_axis);
            let slices_ref: Vec<Vec<i32>> = (0..shape[src_axis])
                .map(|i| ref_slice_at(&data, &shape, src_axis, i))
                .collect();
            for insert_axis in 0..3 {
                let da = stack(&sa, insert_axis);
                let (want_shape, want) = ref_stack(&slices_ref, &slice_shape, insert_axis);
                assert_eq!(da.shape(), want_shape.as_slice());
                assert_eq_slices(
                    &format!("stack3({src_axis}->{insert_axis}) {shape:?}"),
                    da.data(),
                    &want,
                );
            }
        }
    }
}

#[test]
fn stack_rank0_slices() {
    let data = make_i32(&[4], 1);
    let a = Ndarr::new(&data, [4]).unwrap();
    let sa = a.slice_at(0);
    let da = stack(&sa, 0);
    assert_eq!(da.shape(), &[4]);
    assert_eq_slices("stack_rank0_identity", da.data(), &data);
}

#[test]
fn scalar_of_rank0_arrays() {
    for v in [-7i32, 0, 13] {
        let a = rank0(v);
        assert_eq!(a.scalar(), v);
    }
    // Reached through a reduction all the way down to rank 0.
    let data = make_i32(&[6], 2);
    let a = Ndarr::new(&data, [6]).unwrap();
    let ra = a.reduce(0, |x, y| x + y).unwrap();
    assert_eq!(ra.scalar(), data.iter().sum::<i32>());
}

#[test]
fn assign_ops_array_and_scalar() {
    let shape = [2usize, 3];
    let d1 = make_i32(&shape, 1);
    let d2 = make_i32(&shape, 10);
    let mut a = Ndarr::new(&d1, shape).unwrap();
    let c = Ndarr::new(&d2, shape).unwrap();

    a += &c;
    let want: Vec<i32> = d1.iter().zip(&d2).map(|(x, y)| x + y).collect();
    assert_eq_slices("add_assign_array", a.data(), &want);

    a -= &c;
    assert_eq_slices("sub_assign_array", a.data(), &d1);

    a += 3;
    let want: Vec<i32> = d1.iter().map(|x| x + 3).collect();
    assert_eq_slices("add_assign_scalar", a.data(), &want);

    a *= 2;
    let want: Vec<i32> = d1.iter().map(|x| (x + 3) * 2).collect();
    assert_eq_slices("mul_assign_scalar", a.data(), &want);
}

#[test]
fn assign_ops_shape_mismatch_panics() {
    let mut c = Ndarr::new(&make_i32(&[2, 3], 0), [2, 3]).unwrap();
    let e = Ndarr::new(&make_i32(&[3, 2], 0), [3, 2]).unwrap();
    assert!(panics(move || {
        c += &e;
    }));
}

#[test]
fn view_chain_transpose_of_broadcast_of_axis_index() {
    let shape = [2usize, 3, 4];
    let data = make_i32(&shape, 1);
    let a = Ndarr::new(&data, shape).unwrap();
    let target = [2usize, 2, 3];
    let tdim = Dim::<U3>::new(&target).unwrap();

    for index in 0..4 {
        // axis 2 of a C-order array yields a non-contiguous view.
        let v = a.index_axis_view(2, index).unwrap();
        assert!(!v.is_standard_layout());
        let bv = v.broadcast_view_to(&tdim).unwrap();
        assert!(bv.strides().contains(&0));
        let tv = bv.t_view();
        let materialized = tv.to_owned_array();

        let slice = ref_slice_at(&data, &shape, 2, index);
        let broadcast = ref_broadcast(&slice, &[2, 3], &target);
        let (want_shape, want) = ref_transpose(&broadcast, &target);
        assert_eq!(materialized.shape(), want_shape.as_slice());
        assert_eq_slices(&format!("view_chain[{index}]"), materialized.data(), &want);

        let eager = a.slice_at(2)[index].broadcast(&tdim).unwrap().t();
        assert_eq_slices(&format!("eager_chain[{index}]"), eager.data(), &want);
    }
}

#[test]
fn view_of_transposed_broadcast_feeds_further_ops() {
    let shape = [2usize, 3, 4];
    let data = make_i32(&shape, 5);
    let a = Ndarr::new(&data, shape).unwrap();

    let ta = a.t();
    let (tshape, tdata) = ref_transpose(&data, &shape);
    for axis in 0..3 {
        let ra = ta.reduce(axis, |x, y| x + y).unwrap();
        let (want_shape, want) = ref_reduce(&tdata, &tshape, axis, |x, y| x + y);
        assert_eq!(ra.shape(), want_shape.as_slice());
        assert_eq_slices(&format!("reduce_of_t({axis})"), ra.data(), &want);
        assert_eq_slices(
            &format!("roll_of_t({axis})"),
            ta.roll(-3, axis).data(),
            &ref_roll(&tdata, &tshape, -3, axis),
        );
    }
    assert_eq_slices("reshape_of_t", ta.reshape([6, 4]).unwrap().data(), &tdata);

    let sa = a.slice_at(1);
    let dim = Dim::<U3>::new(&[3, 2, 4]).unwrap();
    for (i, sub) in sa.iter().enumerate() {
        let slice = ref_slice_at(&data, &shape, 1, i);
        let broadcast = ref_broadcast(&slice, &[2, 4], &[3, 2, 4]);
        let (bshape, bdata) = ref_transpose(&broadcast, &[3, 2, 4]);
        let ba = sub.broadcast(&dim).unwrap().t();
        assert_eq!(ba.shape(), bshape.as_slice());
        assert_eq_slices(&format!("t_of_broadcast[{i}]"), ba.data(), &bdata);
        for axis in 0..3 {
            let ra = ba.reduce(axis, |x, y| x + y).unwrap();
            let (_, want) = ref_reduce(&bdata, &bshape, axis, |x, y| x + y);
            assert_eq_slices(
                &format!("reduce_of_t_of_broadcast[{i}]({axis})"),
                ra.data(),
                &want,
            );
        }
    }
}

#[test]
fn broadcast_view_zero_strides_and_element_repetition() {
    let a = Ndarr::new(&[1, 2, 3], [3, 1]).unwrap();
    let dim = Dim::<U2>::new(&[3, 4]).unwrap();
    let v = a.broadcast_view_to(&dim).unwrap();
    assert_eq!(v.strides(), &[1, 0]);
    let collected: Vec<i32> = v.iter_elems().cloned().collect();
    assert_eq_slices(
        "zero_stride_iteration",
        &collected,
        &[1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
    );
    for i in 0..3 {
        for j in 0..4 {
            assert_eq!(v[[i, j]], (i as i32) + 1);
        }
    }
    assert!(v.flat_index(&[3, 0]).is_err());
    assert!(v.flat_index(&[0]).is_err());
}

#[test]
fn random_api_differential_i32() {
    let mut g = Lcg::new(0xAD17_0FFE_0042);
    for trial in 0..24 {
        let shape = [
            g.usize_range(1, 4),
            g.usize_range(1, 4),
            g.usize_range(1, 4),
        ];
        let n = product(&shape);
        let data: Vec<i32> = (0..n).map(|_| g.i32_range(-50, 49)).collect();
        let a = Ndarr::new(&data, shape).unwrap();
        let tag = |op: &str| format!("rand[{trial}] {shape:?} {op}");

        let (tshape, tdata) = ref_transpose(&data, &shape);
        assert_eq!(a.t().shape(), tshape.as_slice());
        assert_eq_slices(&tag("t"), a.t().data(), &tdata);
        assert_eq_slices(
            &tag("reshape_flat"),
            a.clone().reshape([n]).unwrap().data(),
            &data,
        );
        assert_eq_slices(
            &tag("reshape_view_flat"),
            a.view().reshape([n]).unwrap().to_owned_array().data(),
            &data,
        );

        for axis in 0..3 {
            let sa = a.slice_at(axis);
            assert_eq!(sa.len(), shape[axis]);
            let mut slice_shape = shape.to_vec();
            slice_shape.remove(axis);
            let slices_ref: Vec<Vec<i32>> = (0..shape[axis])
                .map(|i| ref_slice_at(&data, &shape, axis, i))
                .collect();
            for (i, x) in sa.iter().enumerate() {
                assert_eq_slices(
                    &tag(&format!("slice({axis})[{i}]")),
                    x.data(),
                    &slices_ref[i],
                );
            }
            let insert = g.usize_range(0, 2);
            let da = stack(&sa, insert);
            let (want_shape, want) = ref_stack(&slices_ref, &slice_shape, insert);
            assert_eq!(da.shape(), want_shape.as_slice());
            assert_eq_slices(&tag(&format!("stack({axis}->{insert})")), da.data(), &want);

            let shift = g.isize_range(-7, 7);
            assert_eq_slices(
                &tag(&format!("roll({shift},{axis})")),
                a.roll(shift, axis).data(),
                &ref_roll(&data, &shape, shift, axis),
            );

            let ra = a.reduce(axis, |x, y| x - 2 * y).unwrap();
            let (rshape, rdata) = ref_reduce(&data, &shape, axis, |x, y| x - 2 * y);
            assert_eq!(ra.shape(), rshape.as_slice());
            assert_eq_slices(&tag(&format!("reduce({axis})")), ra.data(), &rdata);
        }

        let target = [g.usize_range(1, 3), shape[0], shape[1], shape[2]];
        let tdim = Dim::<Dyn>::new(&target).unwrap();
        assert_eq_slices(
            &tag("broadcast_padded"),
            a.broadcast(&tdim).unwrap().data(),
            &ref_broadcast(&data, &shape, &target),
        );
    }
}

#[test]
fn random_broadcast_differential() {
    let mut g = Lcg::new(0xB40A_DCA5_7001);
    for trial in 0..32 {
        let d0 = g.usize_range(1, 3);
        let d1 = g.usize_range(1, 4);
        // Source axes are independently collapsed to 1 so 0-strides appear in the middle.
        let s0 = if g.next_u64() % 2 == 0 { 1 } else { d0 };
        let s1 = if g.next_u64() % 2 == 0 { 1 } else { d1 };
        let src = [s0, s1];
        let target = [d0, d1];
        let data: Vec<i32> = (0..product(&src)).map(|_| g.i32_range(-30, 29)).collect();
        let a = Ndarr::new(&data, src).unwrap();
        let tag = |op: &str| format!("rand_bc[{trial}] {src:?}->{target:?} {op}");
        let want = ref_broadcast(&data, &src, &target);

        let ba = a.broadcast(target).unwrap();
        assert_eq_slices(&tag("broadcast"), ba.data(), &want);

        let dim = Dim::<U2>::new(&target).unwrap();
        let view = a.broadcast_view_to(&dim).unwrap();
        assert_eq_slices(
            &tag("broadcast_view_to"),
            view.to_owned_array().data(),
            &want,
        );
        for i in 0..target[0] {
            for j in 0..target[1] {
                let expected =
                    data[(if s0 == 1 { 0 } else { i }) * s1 + if s1 == 1 { 0 } else { j }];
                assert_eq!(view[[i, j]], expected, "{}", tag("index"));
            }
        }

        let (_, twant) = ref_transpose(&want, &target);
        assert_eq_slices(&tag("t_of_broadcast"), ba.t().data(), &twant);
    }
}

#[test]
fn random_float_reduce_and_roll() {
    let mut g = Lcg::new(0xF10A_7002);
    for trial in 0..24 {
        let shape = [g.usize_range(1, 4), g.usize_range(1, 5)];
        let data = make_f64_lcg(product(&shape), &mut g);
        let a = Ndarr::new(&data, shape).unwrap();
        for axis in 0..2 {
            let ra = a.reduce(axis, |x, y| x + y).unwrap();
            let (_, want) = ref_reduce(&data, &shape, axis, |x, y| x + y);
            assert_close(
                &format!("rand_f64_reduce[{trial}]({axis})"),
                ra.data(),
                &want,
                1e-12,
            );
            let shift = g.isize_range(-5, 5);
            assert_close(
                &format!("rand_f64_roll[{trial}]({shift},{axis})"),
                a.roll(shift, axis).data(),
                &ref_roll(&data, &shape, shift, axis),
                0.0,
            );
        }
        let (_, twant) = ref_transpose(&data, &shape);
        assert_close(&format!("rand_f64_t[{trial}]"), a.t().data(), &twant, 0.0);
    }
}
