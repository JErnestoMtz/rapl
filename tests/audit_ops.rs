//! Operators, maps, scans and products, checked against golden values and reference implementations.

use rapl::{
    s, Dim, Dyn, InsertAxisRank, Inserted, Ndarr, Rank, RemoveAxisRank, Removed, ScanDirection, U0,
    U1, U2, U3, U4,
};
use std::fmt::Debug;

/// Reference `scanr`/`scanl`; `f` takes `(element, accumulator)`, the reverse of `scan_axis`.
trait RefScans<T: Clone, R: Rank> {
    fn scanr(&self, axis: usize, f: impl Fn(T, T) -> T) -> Ndarr<T, R>;
    fn scanl(&self, axis: usize, f: impl Fn(T, T) -> T) -> Ndarr<T, R>;
}

/// Owned axis slices built from borrowed axis views.
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

impl<T: Clone, R: Rank> RefScans<T, R> for Ndarr<T, R> {
    fn scanr(&self, axis: usize, f: impl Fn(T, T) -> T) -> Ndarr<T, R> {
        self.scan_axis(axis, ScanDirection::Forward, |acc, value| f(value, acc))
            .expect("axis out of bounds")
    }

    fn scanl(&self, axis: usize, f: impl Fn(T, T) -> T) -> Ndarr<T, R> {
        self.scan_axis(axis, ScanDirection::Backward, |acc, value| f(value, acc))
            .expect("axis out of bounds")
    }
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

/// Deterministic SplitMix64 PRNG.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `lo..=hi`.
    fn gen_usize(&mut self, lo: usize, hi: usize) -> usize {
        lo + (self.next_u64() % (hi - lo + 1) as u64) as usize
    }

    /// Uniform in `lo..=hi`.
    fn gen_isize(&mut self, lo: isize, hi: isize) -> isize {
        lo + (self.next_u64() % (hi - lo + 1) as u64) as isize
    }

    fn gen_bool(&mut self) -> bool {
        self.next_u64() & 1 == 1
    }

    fn gen_f64(&mut self, lo: f64, hi: f64) -> f64 {
        let unit = (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
        lo + (hi - lo) * unit
    }
}

fn rng() -> Rng {
    Rng::new(0x0000_AD17_5EED_0FF5)
}

const N_TRIALS: usize = 32;

fn product(shape: &[usize]) -> usize {
    shape.iter().product()
}

/// Deterministic, sign-varying, never-zero integers (safe for `/` and `%`).
fn ints(shape: &[usize], start: i32) -> Vec<i32> {
    (0..product(shape) as i32)
        .map(|i| {
            let v = start + i;
            let v = if v == 0 { start + i + 7 } else { v };
            if i % 3 == 2 {
                -v
            } else {
                v
            }
        })
        .collect()
}

/// Sequential integers, `start..`.
fn seq(shape: &[usize], start: i32) -> Vec<i32> {
    (0..product(shape) as i32).map(|i| start + i).collect()
}

/// Random floats in (-5, 5) bounded away from zero.
fn floats(shape: &[usize], g: &mut Rng) -> Vec<f64> {
    (0..product(shape))
        .map(|_| {
            let v = g.gen_f64(-5.0, 5.0);
            if v.abs() < 1e-3 {
                v + 1.0
            } else {
                v
            }
        })
        .collect()
}

fn mk<T: Clone, R: Rank>(data: &[T], shape: &[usize]) -> Ndarr<T, R> {
    Ndarr::new(data, Dim::<R>::new(shape).unwrap()).unwrap()
}

fn mk_dyn<T: Clone>(data: &[T], shape: &[usize]) -> Ndarr<T, Dyn> {
    Ndarr::new(data, Dim::<Dyn>::new(shape).unwrap()).unwrap()
}

fn assert_arr<T: PartialEq + Debug + Clone, R: Rank>(
    op: &str,
    arr: &Ndarr<T, R>,
    shape: &[usize],
    expected: &[T],
) {
    assert_eq!(arr.shape(), shape, "{op}: shape mismatch");
    assert_eq!(arr.data().len(), expected.len(), "{op}: length mismatch");
    for (i, (a, b)) in arr.data().iter().zip(expected.iter()).enumerate() {
        assert!(
            a == b,
            "{op}: mismatch at flat index {i} shape={shape:?}: got {a:?} expected {b:?}"
        );
    }
}

fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
    (a - b).abs() <= eps
        || (a.is_nan() && b.is_nan())
        || (a.is_infinite() && b.is_infinite() && a.is_sign_positive() == b.is_sign_positive())
}

fn assert_arr_approx<R: Rank>(
    op: &str,
    arr: &Ndarr<f64, R>,
    shape: &[usize],
    expected: &[f64],
    eps: f64,
) {
    assert_eq!(arr.shape(), shape, "{op}: shape mismatch");
    assert_eq!(arr.data().len(), expected.len(), "{op}: length mismatch");
    for (i, (a, b)) in arr.data().iter().zip(expected.iter()).enumerate() {
        assert!(
            approx_eq(*a, *b, eps),
            "{op}: approx mismatch at {i} shape={shape:?}: got {a:?} expected {b:?}"
        );
    }
}

/// Flat C-order position of a multi-index.
fn flat(shape: &[usize], idx: &[usize]) -> usize {
    let mut pos = 0;
    for (i, &x) in idx.iter().enumerate() {
        pos = pos * shape[i] + x;
    }
    pos
}

/// Call `f` with every multi-index of `shape` in logical C-order.
fn for_each_index(shape: &[usize], mut f: impl FnMut(&[usize])) {
    let n = product(shape);
    let mut idx = vec![0usize; shape.len()];
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

/// NumPy-style co-broadcast shape (a zero-length axis wins against length 1).
fn bshape(sa: &[usize], sb: &[usize]) -> Option<Vec<usize>> {
    let r = sa.len().max(sb.len());
    let mut out = Vec::with_capacity(r);
    for i in 0..r {
        let a = if i + sa.len() >= r {
            sa[i + sa.len() - r]
        } else {
            1
        };
        let b = if i + sb.len() >= r {
            sb[i + sb.len() - r]
        } else {
            1
        };
        let d = if a == b {
            a
        } else if a == 1 {
            b
        } else if b == 1 {
            a
        } else {
            return None;
        };
        out.push(d);
    }
    Some(out)
}

/// Map an output multi-index back onto shape `s` (right-aligned; length-1 axes pinned to 0).
fn project(out_rank: usize, idx: &[usize], s: &[usize]) -> Vec<usize> {
    let off = out_rank - s.len();
    s.iter()
        .enumerate()
        .map(|(i, &extent)| if extent == 1 { 0 } else { idx[off + i] })
        .collect()
}

/// Reference broadcasting element-wise combination.
fn ref_zip<A: Clone, B: Clone, C>(
    sa: &[usize],
    da: &[A],
    sb: &[usize],
    db: &[B],
    f: impl Fn(A, B) -> C,
) -> (Vec<usize>, Vec<C>) {
    let shape = bshape(sa, sb).expect("shapes must co-broadcast");
    let mut out = Vec::with_capacity(product(&shape));
    for_each_index(&shape, |idx| {
        let ia = project(shape.len(), idx, sa);
        let ib = project(shape.len(), idx, sb);
        out.push(f(da[flat(sa, &ia)].clone(), db[flat(sb, &ib)].clone()));
    });
    (shape, out)
}

/// Reference broadcast of one operand to a target shape.
fn ref_broadcast<T: Clone>(s: &[usize], d: &[T], target: &[usize]) -> Vec<T> {
    let mut out = Vec::with_capacity(product(target));
    for_each_index(target, |idx| {
        let i = project(target.len(), idx, s);
        out.push(d[flat(s, &i)].clone());
    });
    out
}

/// Reference full transpose (all axes reversed).
fn ref_transpose<T: Clone>(shape: &[usize], data: &[T]) -> (Vec<usize>, Vec<T>) {
    let mut ts: Vec<usize> = shape.to_vec();
    ts.reverse();
    let mut out = Vec::with_capacity(data.len());
    for_each_index(&ts, |idx| {
        let src: Vec<usize> = idx.iter().rev().copied().collect();
        out.push(data[flat(shape, &src)].clone());
    });
    (ts, out)
}

/// Reference rank-lowering axis selection (`slice_at(axis)[index]`).
fn ref_index_axis<T: Clone>(
    shape: &[usize],
    data: &[T],
    axis: usize,
    index: usize,
) -> (Vec<usize>, Vec<T>) {
    let mut os = shape.to_vec();
    os.remove(axis);
    let mut out = Vec::with_capacity(product(&os));
    for_each_index(&os, |idx| {
        let mut full = idx.to_vec();
        full.insert(axis, index);
        out.push(data[flat(shape, &full)].clone());
    });
    (os, out)
}

/// Reference roll: element at axis position `j` moves to `j + shift` (mod len).
fn ref_roll<T: Clone>(shape: &[usize], data: &[T], shift: isize, axis: usize) -> Vec<T> {
    let len = shape[axis];
    if len == 0 {
        return data.to_vec();
    }
    let shift = shift.rem_euclid(len as isize) as usize;
    let mut out = Vec::with_capacity(data.len());
    for_each_index(shape, |idx| {
        let mut src = idx.to_vec();
        src[axis] = (idx[axis] + len - shift) % len;
        out.push(data[flat(shape, &src)].clone());
    });
    out
}

/// Reference reduce: left fold along `axis` (`g(g(x0, x1), x2)...`).
fn ref_reduce<T: Clone>(
    shape: &[usize],
    data: &[T],
    axis: usize,
    g: impl Fn(T, T) -> T,
) -> (Vec<usize>, Vec<T>) {
    let mut os = shape.to_vec();
    os.remove(axis);
    let mut out = Vec::with_capacity(product(&os));
    for_each_index(&os, |idx| {
        let mut acc: Option<T> = None;
        for k in 0..shape[axis] {
            let mut full = idx.to_vec();
            full.insert(axis, k);
            let v = data[flat(shape, &full)].clone();
            acc = Some(match acc {
                None => v,
                Some(a) => g(a, v),
            });
        }
        out.push(acc.expect("reduced axis is non-empty"));
    });
    (os, out)
}

/// Reference `scanr`: `out[0] = x[0]`, `out[k] = f(x[k], out[k-1])`.
fn ref_scanr<T: Clone>(shape: &[usize], data: &[T], axis: usize, f: impl Fn(T, T) -> T) -> Vec<T> {
    let len = shape[axis];
    let mut frame = shape.to_vec();
    frame.remove(axis);
    let mut out: Vec<Option<T>> = vec![None; data.len()];
    for k in 0..len {
        for_each_index(&frame, |idx| {
            let mut full = idx.to_vec();
            full.insert(axis, k);
            let cur = data[flat(shape, &full)].clone();
            let v = if k == 0 {
                cur
            } else {
                let mut prev = full.clone();
                prev[axis] = k - 1;
                f(cur, out[flat(shape, &prev)].clone().unwrap())
            };
            out[flat(shape, &full)] = Some(v);
        });
    }
    out.into_iter().map(|x| x.unwrap()).collect()
}

/// Reference `scanl`: `out[last] = x[last]`, `out[k] = f(x[k], out[k+1])`.
fn ref_scanl<T: Clone>(shape: &[usize], data: &[T], axis: usize, f: impl Fn(T, T) -> T) -> Vec<T> {
    let len = shape[axis];
    let mut frame = shape.to_vec();
    frame.remove(axis);
    let mut out: Vec<Option<T>> = vec![None; data.len()];
    for k in (0..len).rev() {
        for_each_index(&frame, |idx| {
            let mut full = idx.to_vec();
            full.insert(axis, k);
            let cur = data[flat(shape, &full)].clone();
            let v = if k == len - 1 {
                cur
            } else {
                let mut next = full.clone();
                next[axis] = k + 1;
                f(cur, out[flat(shape, &next)].clone().unwrap())
            };
            out[flat(shape, &full)] = Some(v);
        });
    }
    out.into_iter().map(|x| x.unwrap()).collect()
}

/// Reference `inner_product`: contract a's last axis with b's first (length-1 axes broadcast), pair with `f`, left-fold with `g`.
fn ref_inner<A: Clone, B: Clone, C: Clone>(
    sa: &[usize],
    da: &[A],
    sb: &[usize],
    db: &[B],
    f: impl Fn(A, B) -> C,
    g: impl Fn(C, C) -> C,
) -> (Vec<usize>, Vec<C>) {
    let ka = sa[sa.len() - 1];
    let kb = sb[0];
    assert!(
        ka == kb || ka == 1 || kb == 1,
        "contraction axes must match or broadcast"
    );
    let k_len = ka.max(kb);
    let mut os: Vec<usize> = sa[..sa.len() - 1].to_vec();
    os.extend_from_slice(&sb[1..]);
    let mut out = Vec::with_capacity(product(&os));
    for_each_index(&os, |idx| {
        let (i_, j_) = idx.split_at(sa.len() - 1);
        let mut acc: Option<C> = None;
        for k in 0..k_len {
            let mut ia = i_.to_vec();
            ia.push(if ka == 1 { 0 } else { k });
            let mut ib = vec![if kb == 1 { 0 } else { k }];
            ib.extend_from_slice(j_);
            let v = f(da[flat(sa, &ia)].clone(), db[flat(sb, &ib)].clone());
            acc = Some(match acc {
                None => v,
                Some(a) => g(a, v),
            });
        }
        out.push(acc.expect("contraction axis is non-empty"));
    });
    (os, out)
}

/// Reference matmul: `ref_inner` with multiply/add.
fn ref_matmul(sa: &[usize], da: &[i32], sb: &[usize], db: &[i32]) -> (Vec<usize>, Vec<i32>) {
    ref_inner(sa, da, sb, db, |x, y| x * y, |x, y| x + y)
}

/// Reference outer product: shape `sa ++ sb`, element `f(a[i], b[j])`.
fn ref_outer<A: Clone, B: Clone, C>(
    sa: &[usize],
    da: &[A],
    sb: &[usize],
    db: &[B],
    f: impl Fn(A, B) -> C,
) -> (Vec<usize>, Vec<C>) {
    let mut os = sa.to_vec();
    os.extend_from_slice(sb);
    let mut out = Vec::with_capacity(product(&os));
    for_each_index(&os, |idx| {
        let (i_, j_) = idx.split_at(sa.len());
        out.push(f(da[flat(sa, i_)].clone(), db[flat(sb, j_)].clone()));
    });
    (os, out)
}

macro_rules! dyadic_case {
    ($name:expr, $ra:ty, $sa:expr, $rb:ty, $sb:expr) => {{
        let da = ints($sa, 1);
        let db = ints($sb, 5);
        let a = mk::<i32, $ra>(&da, $sa);
        let b = mk::<i32, $rb>(&db, $sb);
        for (tag, f) in [
            ("add", (|x, y| x + y) as fn(i32, i32) -> i32),
            ("sub", |x, y| x - y),
            ("mul", |x, y| x * y),
            ("div", |x, y| x / y),
            ("rem", |x, y| x % y),
        ] {
            let (fs, fd) = ref_zip($sa, &da, $sb, &db, f);
            assert_arr(
                &format!("{}/{tag}/fwd", $name),
                &a.zip_with(&b, |x, y| f(*x, *y)).unwrap(),
                &fs,
                &fd,
            );
            let (rs, rd) = ref_zip($sb, &db, $sa, &da, f);
            assert_arr(
                &format!("{}/{tag}/rev", $name),
                &b.zip_with(&a, |x, y| f(*x, *y)).unwrap(),
                &rs,
                &rd,
            );
        }
    }};
}

#[test]
fn zip_with_mixed_rank_asymmetric() {
    dyadic_case!("[2,3,4]x[4]", U3, &[2, 3, 4], U1, &[4]);
    dyadic_case!("[2,3,4]x[3,4]", U3, &[2, 3, 4], U2, &[3, 4]);
    dyadic_case!("[2,3,4]x[1,4]", U3, &[2, 3, 4], U2, &[1, 4]);
    dyadic_case!("[2,3,4]x[3,1]", U3, &[2, 3, 4], U2, &[3, 1]);
    dyadic_case!("[3,1,4]x[5,4]", U3, &[3, 1, 4], U2, &[5, 4]);
    dyadic_case!("[3,1,4]x[1,5,1]", U3, &[3, 1, 4], U3, &[1, 5, 1]);
}

#[test]
fn zip_with_cobroadcast_row_col() {
    dyadic_case!("[1,5]x[5,1]", U2, &[1, 5], U2, &[5, 1]);
    dyadic_case!("[5,1]x[1,5]", U2, &[5, 1], U2, &[1, 5]);
    dyadic_case!("[1,1]x[4,3]", U2, &[1, 1], U2, &[4, 3]);
    dyadic_case!("[7]x[7,1]", U1, &[7], U2, &[7, 1]);
}

#[test]
fn zip_with_len1_and_single_element() {
    dyadic_case!("[1]x[1]", U1, &[1], U1, &[1]);
    dyadic_case!("[1]x[2,3,4]", U1, &[1], U3, &[2, 3, 4]);
    dyadic_case!("[1,1,1]x[2,3,4]", U3, &[1, 1, 1], U3, &[2, 3, 4]);
    dyadic_case!("[1,1]x[1]", U2, &[1, 1], U1, &[1]);
}

#[test]
fn zip_with_rank0() {
    let s = mk::<i32, U0>(&[9], &[]);
    let da = ints(&[2, 3], 1);
    let a = mk::<i32, U2>(&da, &[2, 3]);

    let (fs, fd) = ref_zip(&[], &[9], &[2, 3], &da, |x, y| x - y);
    assert_arr(
        "rank0_sub_rank2",
        &s.zip_with(&a, |x, y| x - y).unwrap(),
        &fs,
        &fd,
    );
    let (rs, rd) = ref_zip(&[2, 3], &da, &[], &[9], |x, y| x - y);
    assert_arr(
        "rank2_sub_rank0",
        &a.zip_with(&s, |x, y| x - y).unwrap(),
        &rs,
        &rd,
    );

    let t = mk::<i32, U0>(&[4], &[]);
    assert_arr(
        "rank0_x_rank0",
        &s.zip_with(&t, |x, y| x - y).unwrap(),
        &[],
        &[5],
    );
}

#[test]
fn zip_with_incompatible_is_error_not_panic() {
    let a = mk::<i32, U2>(&ints(&[2, 3], 1), &[2, 3]);
    let b = mk::<i32, U1>(&ints(&[4], 1), &[4]);
    assert!(a.zip_with(&b, |x, y| x + y).is_err());
    assert!(b.zip_with(&a, |x, y| x + y).is_err());

    let c = mk::<i32, U3>(&ints(&[2, 3, 4], 1), &[2, 3, 4]);
    let d = mk::<i32, U2>(&ints(&[2, 4], 1), &[2, 4]);
    assert!(c.zip_with(&d, |x, y| x + y).is_err());
}

#[test]
fn zip_with_zero_axis() {
    let a = mk::<i32, U2>(&[], &[0, 3]);
    let b = mk::<i32, U2>(&[], &[0, 3]);
    let r = a.zip_with(&b, |x, y| x + y).unwrap();
    assert_arr("zero_axis_same_shape", &r, &[0, 3], &[]);
    assert!(r.data().is_empty());

    let c = mk::<i32, U2>(&[], &[3, 0]);
    assert_arr(
        "zero_trailing_axis",
        &c.zip_with(&c, |x, y| x * y).unwrap(),
        &[3, 0],
        &[],
    );
}

#[test]
fn zip_with_heterogeneous_types() {
    let da = ints(&[3, 4], 1);
    let db = ints(&[4], 2);
    let a = mk::<i32, U2>(&da, &[3, 4]);
    let b = mk::<i32, U1>(&db, &[4]);
    let f = |x: i32, y: i32| (x > y, x as f64 / y as f64);
    let r = a.zip_with(&b, |x, y| f(*x, *y)).unwrap();
    let (es, ed) = ref_zip(&[3, 4], &da, &[4], &db, f);
    assert_eq!(r.shape(), es.as_slice());
    for (i, (x, y)) in r.data().iter().zip(ed.iter()).enumerate() {
        assert_eq!(x.0, y.0, "hetero bool at {i}");
        assert!((x.1 - y.1).abs() < 1e-12, "hetero f64 at {i}");
    }
    let g = |x: i32, y: i32| format!("{x}:{y}");
    let (gs, gd) = ref_zip(&[4], &db, &[3, 4], &da, g);
    assert_arr(
        "hetero_string",
        &b.zip_with(&a, |x, y| g(*x, *y)).unwrap(),
        &gs,
        &gd,
    );
}

#[test]
fn zip_with_random_shapes() {
    let mut g = rng();
    for trial in 0..N_TRIALS {
        let d0 = g.gen_usize(1, 3);
        let d1 = g.gen_usize(1, 4);
        let d2 = g.gen_usize(1, 4);
        let e1 = if g.gen_bool() { 1 } else { d1 };
        let e2 = if g.gen_bool() { 1 } else { d2 };
        let sa = [d0, d1, d2];
        let sb = [e1, e2];
        let da = ints(&sa, trial as i32 + 1);
        let db = ints(&sb, trial as i32 + 31);
        let a = mk::<i32, U3>(&da, &sa);
        let b = mk::<i32, U2>(&db, &sb);

        let (s1, r1) = ref_zip(&sa, &da, &sb, &db, |x, y| x - y);
        assert_arr(
            &format!("rand_dyadic_sub[{trial}]"),
            &a.zip_with(&b, |x, y| x - y).unwrap(),
            &s1,
            &r1,
        );
        let (s2, r2) = ref_zip(&sb, &db, &sa, &da, |x, y| x - y);
        assert_arr(
            &format!("rand_dyadic_sub_rev[{trial}]"),
            &b.zip_with(&a, |x, y| x - y).unwrap(),
            &s2,
            &r2,
        );
        let (s3, r3) = ref_zip(&sa, &da, &sb, &db, |x, y| x / y);
        assert_arr(
            &format!("rand_dyadic_div[{trial}]"),
            &a.zip_with(&b, |x, y| x / y).unwrap(),
            &s3,
            &r3,
        );
    }
}

macro_rules! all_operator_forms {
    ($name:expr, $ra:ty, $sa:expr, $rb:ty, $sb:expr) => {{
        let da = ints($sa, 3);
        let db = ints($sb, 11);
        let a = mk::<i32, $ra>(&da, $sa);
        let b = mk::<i32, $rb>(&db, $sb);

        let (s_add, d_add) = ref_zip($sa, &da, $sb, &db, |x, y| x + y);
        assert_arr(
            &format!("{}/ref_ref/add", $name),
            &(&a + &b),
            &s_add,
            &d_add,
        );
        assert_arr(
            &format!("{}/own_ref/add", $name),
            &(a.clone() + &b),
            &s_add,
            &d_add,
        );
        assert_arr(
            &format!("{}/ref_own/add", $name),
            &(&a + b.clone()),
            &s_add,
            &d_add,
        );
        assert_arr(
            &format!("{}/own_own/add", $name),
            &(a.clone() + b.clone()),
            &s_add,
            &d_add,
        );

        let (s_sub, d_sub) = ref_zip($sa, &da, $sb, &db, |x, y| x - y);
        assert_arr(
            &format!("{}/ref_ref/sub", $name),
            &(&a - &b),
            &s_sub,
            &d_sub,
        );
        assert_arr(
            &format!("{}/own_ref/sub", $name),
            &(a.clone() - &b),
            &s_sub,
            &d_sub,
        );
        assert_arr(
            &format!("{}/ref_own/sub", $name),
            &(&a - b.clone()),
            &s_sub,
            &d_sub,
        );
        assert_arr(
            &format!("{}/own_own/sub", $name),
            &(a.clone() - b.clone()),
            &s_sub,
            &d_sub,
        );
        let (s_subr, d_subr) = ref_zip($sb, &db, $sa, &da, |x, y| x - y);
        assert_arr(
            &format!("{}/ref_ref/sub_swapped", $name),
            &(&b - &a),
            &s_subr,
            &d_subr,
        );

        let (s_mul, d_mul) = ref_zip($sa, &da, $sb, &db, |x, y| x * y);
        assert_arr(
            &format!("{}/ref_ref/mul", $name),
            &(&a * &b),
            &s_mul,
            &d_mul,
        );
        assert_arr(
            &format!("{}/own_own/mul", $name),
            &(a.clone() * b.clone()),
            &s_mul,
            &d_mul,
        );

        let (s_div, d_div) = ref_zip($sa, &da, $sb, &db, |x, y| x / y);
        assert_arr(
            &format!("{}/ref_ref/div", $name),
            &(&a / &b),
            &s_div,
            &d_div,
        );
        let (s_divr, d_divr) = ref_zip($sb, &db, $sa, &da, |x, y| x / y);
        assert_arr(
            &format!("{}/ref_ref/div_swapped", $name),
            &(&b / &a),
            &s_divr,
            &d_divr,
        );
        assert_arr(
            &format!("{}/own_ref/div", $name),
            &(a.clone() / &b),
            &s_div,
            &d_div,
        );

        let (s_rem, d_rem) = ref_zip($sa, &da, $sb, &db, |x, y| x % y);
        assert_arr(
            &format!("{}/ref_ref/rem", $name),
            &(&a % &b),
            &s_rem,
            &d_rem,
        );
        let (s_remr, d_remr) = ref_zip($sb, &db, $sa, &da, |x, y| x % y);
        assert_arr(
            &format!("{}/ref_ref/rem_swapped", $name),
            &(&b % &a),
            &s_remr,
            &d_remr,
        );
    }};
}

#[test]
fn operator_forms_same_shape() {
    all_operator_forms!("[2,3]", U2, &[2, 3], U2, &[2, 3]);
    all_operator_forms!("[7]", U1, &[7], U1, &[7]);
    all_operator_forms!("[2,3,4]", U3, &[2, 3, 4], U3, &[2, 3, 4]);
}

#[test]
fn operator_forms_broadcast() {
    all_operator_forms!("[2,3,4]x[4]", U3, &[2, 3, 4], U1, &[4]);
    all_operator_forms!("[2,3,4]x[3,1]", U3, &[2, 3, 4], U2, &[3, 1]);
    all_operator_forms!("[1,5]x[5,1]", U2, &[1, 5], U2, &[5, 1]);
    all_operator_forms!("[3,1,4]x[5,4]", U3, &[3, 1, 4], U2, &[5, 4]);
    all_operator_forms!("[1]x[2,3]", U1, &[1], U2, &[2, 3]);
}

#[test]
fn operator_forms_rank0() {
    all_operator_forms!("[]x[]", U0, &[], U0, &[]);
    all_operator_forms!("[]x[3,2]", U0, &[], U2, &[3, 2]);
}

#[test]
fn operator_float() {
    let mut g = rng();
    let sa = [3, 1, 4];
    let sb = [5, 4];
    let da = floats(&sa, &mut g);
    let db = floats(&sb, &mut g);
    let a = mk::<f64, U3>(&da, &sa);
    let b = mk::<f64, U2>(&db, &sb);
    let eps = 1e-12;

    let (s, d) = ref_zip(&sa, &da, &sb, &db, |x, y| x + y);
    assert_arr_approx("f64/add", &(&a + &b), &s, &d, eps);
    let (s, d) = ref_zip(&sa, &da, &sb, &db, |x, y| x - y);
    assert_arr_approx("f64/sub", &(&a - &b), &s, &d, eps);
    let (s, d) = ref_zip(&sb, &db, &sa, &da, |x, y| x - y);
    assert_arr_approx("f64/sub_rev", &(&b - &a), &s, &d, eps);
    let (s, d) = ref_zip(&sa, &da, &sb, &db, |x, y| x * y);
    assert_arr_approx("f64/mul", &(&a * &b), &s, &d, eps);
    let (s, d) = ref_zip(&sa, &da, &sb, &db, |x, y| x / y);
    assert_arr_approx("f64/div", &(&a / &b), &s, &d, eps);
    let (s, d) = ref_zip(&sb, &db, &sa, &da, |x, y| x / y);
    assert_arr_approx("f64/div_rev", &(&b / &a), &s, &d, eps);
    let (s, d) = ref_zip(&sa, &da, &sb, &db, |x, y| x % y);
    assert_arr_approx("f64/rem", &(&a % &b), &s, &d, eps);
}

#[test]
fn scalar_ops_array_first() {
    let s = [2, 3, 4];
    let da = ints(&s, 1);
    let a = mk::<i32, U3>(&da, &s);
    let map = |f: fn(i32) -> i32| da.iter().map(|&x| f(x)).collect::<Vec<_>>();
    assert_arr("scalar/add/ref", &(&a + 5), &s, &map(|x| x + 5));
    assert_arr("scalar/add/own", &(a.clone() + 5), &s, &map(|x| x + 5));
    assert_arr("scalar/sub/ref", &(&a - 5), &s, &map(|x| x - 5));
    assert_arr("scalar/sub/own", &(a.clone() - 5), &s, &map(|x| x - 5));
    assert_arr("scalar/mul/ref", &(&a * 3), &s, &map(|x| x * 3));
    assert_arr("scalar/div/ref", &(&a / 2), &s, &map(|x| x / 2));
    assert_arr("scalar/rem/ref", &(&a % 3), &s, &map(|x| x % 3));
    assert_arr("scalar/div/neg", &(&a / -2), &s, &map(|x| x / -2));

    let mut g = rng();
    let df = floats(&[3, 5], &mut g);
    let f = mk::<f64, U2>(&df, &[3, 5]);
    let fmap = |h: fn(f64) -> f64| df.iter().map(|&x| h(x)).collect::<Vec<_>>();
    assert_arr_approx(
        "scalar/f64/mul",
        &(&f * 2.5),
        &[3, 5],
        &fmap(|x| x * 2.5),
        1e-12,
    );
    assert_arr_approx(
        "scalar/f64/div",
        &(&f / 0.25),
        &[3, 5],
        &fmap(|x| x / 0.25),
        1e-12,
    );
    assert_arr_approx(
        "scalar/f64/sub",
        &(f.clone() - 1.5),
        &[3, 5],
        &fmap(|x| x - 1.5),
        1e-12,
    );
}

#[test]
fn scalar_ops_scalar_first() {
    // `s op arr` is implemented as `arr op s`, so operand order is swapped.
    let s = [2, 5];
    let da = ints(&s, 2);
    let a = mk::<i32, U2>(&da, &s);
    let map = |f: fn(i32) -> i32| da.iter().map(|&x| f(x)).collect::<Vec<_>>();
    assert_arr("scalar_first/add", &(5 + a.clone()), &s, &map(|x| x + 5));
    assert_arr("scalar_first/add_ref", &(5 + &a), &s, &map(|x| x + 5));
    assert_arr("scalar_first/sub", &(5 - a.clone()), &s, &map(|x| x - 5));
    assert_arr("scalar_first/sub_ref", &(5 - &a), &s, &map(|x| x - 5));
    assert_arr("scalar_first/mul", &(3 * a.clone()), &s, &map(|x| x * 3));
    assert_arr("scalar_first/div", &(2 / a.clone()), &s, &map(|x| x / 2));
    assert_arr("scalar_first/rem", &(3 % a.clone()), &s, &map(|x| x % 3));

    let mut g = rng();
    let df = floats(&[6], &mut g);
    let f = mk::<f64, U1>(&df, &[6]);
    let sub: Vec<f64> = df.iter().map(|&x| x - 2.0).collect();
    assert_arr_approx(
        "scalar_first/f64/sub",
        &(2.0 - f.clone()),
        &[6],
        &sub,
        1e-12,
    );
    let div: Vec<f64> = df.iter().map(|&x| x / 2.0).collect();
    assert_arr_approx("scalar_first/f64/div", &(2.0 / &f), &[6], &div, 1e-12);
}

#[test]
fn neg() {
    let s = [2, 3, 4];
    let da = ints(&s, -7);
    let a = mk::<i32, U3>(&da, &s);
    let expected: Vec<i32> = da.iter().map(|&x| -x).collect();
    assert_arr("neg/own", &(-a.clone()), &s, &expected);
    assert_arr("neg/ref", &(-&a), &s, &expected);

    let mut g = rng();
    let df = floats(&[9], &mut g);
    let f = mk::<f64, U1>(&df, &[9]);
    let fneg: Vec<f64> = df.iter().map(|&x| -x).collect();
    assert_arr_approx("neg/f64", &(-&f), &[9], &fneg, 0.0);

    let z = mk::<i32, U2>(&[], &[0, 4]);
    assert_arr("neg/zero_axis", &(-z), &[0, 4], &[]);
}

#[test]
fn assign_ops_array_rhs() {
    let s = [3, 4];
    let dl = ints(&s, 1);
    let dr = ints(&s, 13);
    let r = mk::<i32, U2>(&dr, &s);
    type BinFn = fn(i32, i32) -> i32;
    let fns: [(&str, BinFn); 5] = [
        ("add", |x, y| x + y),
        ("sub", |x, y| x - y),
        ("mul", |x, y| x * y),
        ("div", |x, y| x / y),
        ("rem", |x, y| x % y),
    ];
    for (i, (tag, f)) in fns.iter().enumerate() {
        let mut a = mk::<i32, U2>(&dl, &s);
        match i {
            0 => a += &r,
            1 => a -= &r,
            2 => a *= &r,
            3 => a /= &r,
            _ => a %= &r,
        }
        let expected: Vec<i32> = dl.iter().zip(dr.iter()).map(|(&x, &y)| f(x, y)).collect();
        assert_arr(&format!("assign/{tag}"), &a, &s, &expected);
    }
}

#[test]
fn assign_ops_scalar_rhs() {
    let s = [2, 3, 4];
    let dl = ints(&s, 1);
    type UnFn = fn(i32) -> i32;
    let fns: [(&str, UnFn); 5] = [
        ("add", |x| x + 6),
        ("sub", |x| x - 6),
        ("mul", |x| x * 6),
        ("div", |x| x / 3),
        ("rem", |x| x % 3),
    ];
    for (i, (tag, f)) in fns.iter().enumerate() {
        let mut a = mk::<i32, U3>(&dl, &s);
        match i {
            0 => a += 6,
            1 => a -= 6,
            2 => a *= 6,
            3 => a /= 3,
            _ => a %= 3,
        }
        let expected: Vec<i32> = dl.iter().map(|&x| f(x)).collect();
        assert_arr(&format!("assign_scalar/{tag}"), &a, &s, &expected);
    }

    let mut g = rng();
    let d = floats(&[4, 3], &mut g);
    let mut a = mk::<f64, U2>(&d, &[4, 3]);
    a *= 0.5;
    let expected: Vec<f64> = d.iter().map(|&x| x * 0.5).collect();
    assert_arr_approx("assign_scalar/f64/mul", &a, &[4, 3], &expected, 1e-12);
}

#[test]
fn assign_ops_on_transposed_receiver() {
    let s = [2, 3, 4];
    let da = ints(&s, 1);
    let a = mk::<i32, U3>(&da, &s);
    // `t()` returns an owned array, so the receiver is the owned transpose.
    let mut ta = a.t();
    let rhs_shape = [4, 3, 2];
    let dr = ints(&rhs_shape, 17);
    let r = mk::<i32, U3>(&dr, &rhs_shape);
    ta -= &r;
    let (ts, td) = ref_transpose(&s, &da);
    let expected: Vec<i32> = td.iter().zip(dr.iter()).map(|(&x, &y)| x - y).collect();
    assert_arr("assign/transposed_receiver", &ta, &ts, &expected);
}

#[test]
fn chained_ops_non_contiguous_sources() {
    let s = [2, 3, 4];
    let da = ints(&s, 1);
    let db = ints(&[3, 4], 9);
    let dc = ints(&[2, 3], 5);
    let de = ints(&[2, 5], 3);
    let a = mk::<i32, U3>(&da, &s);
    let b = mk::<i32, U2>(&db, &[3, 4]);
    let c = mk::<i32, U2>(&dc, &[2, 3]);
    let e = mk::<i32, U2>(&de, &[2, 5]);

    let (tas, tad) = ref_transpose(&s, &da);
    let (tcs, tcd) = ref_transpose(&[2, 3], &dc);
    let (s1, d1) = ref_zip(&tas, &tad, &tcs, &tcd, |x, y| x + y);
    assert_arr("chain/t_then_add", &(&a.t() + &c.t()), &s1, &d1);

    let (sab, dab) = ref_zip(&s, &da, &[3, 4], &db, |x, y| x + y);
    let rolled = ref_roll(&s, &da, 1, 2);
    let (s2, d2) = ref_zip(&sab, &dab, &s, &rolled, |x, y| x - y);
    assert_arr(
        "chain/broadcast_then_sub",
        &(&(&a + &b) - &a.roll(1, 2)),
        &s2,
        &d2,
    );

    let (srs, srd) = ref_reduce(&s, &da, 0, |x, y| x + y);
    let (s3, d3) = ref_zip(&srs, &srd, &[3, 4], &db, |x, y| x / y);
    assert_arr(
        "chain/reduce_then_div",
        &(&a.reduce(0, |x, y| x + y).unwrap() / &b),
        &s3,
        &d3,
    );

    let (sls, sld) = ref_index_axis(&s, &da, 1, 2);
    let (bls, bld) = ref_index_axis(&[3, 4], &db, 0, 1);
    let (s4, d4) = ref_zip(&sls, &sld, &bls, &bld, |x, y| x * y);
    assert_arr(
        "chain/slice_then_mul",
        &(&a.index_axis_view(1, 2).unwrap() * &b.index_axis_view(0, 1).unwrap()),
        &s4,
        &d4,
    );

    let lhs: Vec<i32> = tad.iter().map(|&x| x * 2).collect();
    let rhs: Vec<i32> = de.iter().map(|&x| x + 1).collect();
    let (s5, d5) = ref_matmul(&tas, &lhs, &[2, 5], &rhs);
    assert_arr(
        "chain/scalar_then_matmul",
        &(&a.t() * 2).mat_mul(&(&e + 1)).unwrap(),
        &s5,
        &d5,
    );
}

#[test]
fn map_shapes_and_types() {
    for (name, shape) in [
        ("[]", &[][..]),
        ("[1]", &[1][..]),
        ("[7]", &[7][..]),
        ("[1,5]", &[1, 5][..]),
        ("[5,1]", &[5, 1][..]),
        ("[2,3,4]", &[2, 3, 4][..]),
        ("[0,3]", &[0, 3][..]),
        ("[3,0]", &[3, 0][..]),
    ] {
        let d = ints(shape, 2);
        let a = mk_dyn(&d, shape);
        let ei: Vec<i32> = d.iter().map(|&x| x * 3 - 1).collect();
        assert_arr(
            &format!("map/{name}/i32"),
            &a.map(|x| x * 3 - 1),
            shape,
            &ei,
        );
        let eb: Vec<bool> = d.iter().map(|&x| x > 0).collect();
        assert_arr(&format!("map/{name}/bool"), &a.map(|x| *x > 0), shape, &eb);
        let ef: Vec<f64> = d.iter().map(|&x| x as f64 / 3.0).collect();
        assert_arr(
            &format!("map/{name}/f64"),
            &a.map(|x| *x as f64 / 3.0),
            shape,
            &ef,
        );
    }
}

#[test]
fn map_on_views_matches_eager() {
    let s = [2, 3, 4];
    let da = ints(&s, 1);
    let a = mk::<i32, U3>(&da, &s);

    let (ts, td) = ref_transpose(&s, &da);
    let et: Vec<i32> = td.iter().map(|&x| x * 2).collect();
    assert_arr("map/t_view", &a.t_view().map(|x| x * 2), &ts, &et);
    assert_arr("map/t_owned", &a.t().map(|x| x * 2), &ts, &et);

    let dim = Dim::<U3>::new(&[2, 3, 4]).unwrap();
    let drow = ints(&[4], 5);
    let row = mk::<i32, U1>(&drow, &[4]);
    let bview = row.broadcast_view_to(&dim).unwrap();
    let broadcast = ref_broadcast(&[4], &drow, &s);
    let eb: Vec<i32> = broadcast.iter().map(|&x| x - 1).collect();
    assert_arr("map/broadcast_view_to", &bview.map(|x| x - 1), &s, &eb);

    for (axis, extent) in s.iter().enumerate() {
        for i in 0..*extent {
            let v = a.index_axis_view(axis, i).unwrap();
            let (es, ed) = ref_index_axis(&s, &da, axis, i);
            let expected: Vec<i32> = ed.iter().map(|&x| -x).collect();
            assert_arr(
                &format!("map/index_axis_view({axis},{i})"),
                &v.map(|x| -x),
                &es,
                &expected,
            );
        }
    }

    let tv = a.t_view();
    let nested = tv.index_axis_view(0, 2).unwrap();
    let (ns, nd) = ref_index_axis(&ts, &td, 0, 2);
    let en: Vec<i32> = nd.iter().map(|&x| x + 1000).collect();
    assert_arr("map/nested_view", &nested.map(|x| x + 1000), &ns, &en);
}

#[test]
fn zip_with_owned_and_views() {
    let s = [3, 4];
    let da = ints(&s, 1);
    let db = ints(&s, 21);
    let a = mk::<i32, U2>(&da, &s);
    let b = mk::<i32, U2>(&db, &s);

    let e1: Vec<i32> = da.iter().zip(db.iter()).map(|(&x, &y)| x - y).collect();
    assert_arr(
        "zip_with/owned",
        &a.zip_with(&b, |x, y| x - y).unwrap(),
        &s,
        &e1,
    );
    let e2: Vec<i32> = db.iter().zip(da.iter()).map(|(&x, &y)| x - y).collect();
    assert_arr(
        "zip_with/owned_rev",
        &b.zip_with(&a, |x, y| x - y).unwrap(),
        &s,
        &e2,
    );

    let (ts, tda) = ref_transpose(&s, &da);
    let (_, tdb) = ref_transpose(&s, &db);
    let e3: Vec<i32> = tda.iter().zip(tdb.iter()).map(|(&x, &y)| x - y).collect();
    assert_arr(
        "zip_with/view_view",
        &a.t_view().zip_with(&b.t_view(), |x, y| x - y).unwrap(),
        &ts,
        &e3,
    );
    let ta = a.t();
    let e4: Vec<i32> = tda
        .iter()
        .zip(tdb.iter())
        .map(|(&x, &y)| x * 2 - y)
        .collect();
    assert_arr(
        "zip_with/owned_view",
        &ta.zip_with(&b.t_view(), |x, y| x * 2 - y).unwrap(),
        &ts,
        &e4,
    );
    let dim = Dim::<U2>::new(&s).unwrap();
    let dcol = ints(&[3, 1], 4);
    let col = mk::<i32, U2>(&dcol, &[3, 1]);
    let cview = col.broadcast_view_to(&dim).unwrap();
    let col_full = ref_broadcast(&[3, 1], &dcol, &s);
    let e5: Vec<i32> = da
        .iter()
        .zip(col_full.iter())
        .map(|(&x, &y)| x - y)
        .collect();
    assert_arr(
        "zip_with/broadcast_view_rhs",
        &a.zip_with(&cview, |x, y| x - y).unwrap(),
        &s,
        &e5,
    );
}

#[test]
fn map_in_place_owned_and_view_mut() {
    let s = [2, 3, 4];
    let d = ints(&s, 1);
    let expected: Vec<i32> = d.iter().map(|&x| x * 2 + 1).collect();

    let mut a = mk::<i32, U3>(&d, &s);
    a.map_in_place(|x| x * 2 + 1);
    assert_arr("map_in_place/owned", &a, &s, &expected);

    let mut b = mk::<i32, U3>(&d, &s);
    {
        let mut v = b.view_mut();
        v.map_in_place(|x| x * 2 + 1);
    }
    assert_arr("map_in_place/view_mut_full", &b, &s, &expected);

    let mut c = mk::<i32, U3>(&d, &s);
    {
        let mut sub = c.index_axis_mut(1, 2).unwrap();
        sub.map_in_place(|x| x + 100);
    }
    let mut expected2 = d.clone();
    for i in 0..s[0] {
        for k in 0..s[2] {
            expected2[flat(&s, &[i, 2, k])] += 100;
        }
    }
    assert_arr("map_in_place/index_axis_mut", &c, &s, &expected2);
}

#[test]
fn zip_with_in_place_strided_operands() {
    let s = [3, 4];
    let da = ints(&s, 1);
    let db = ints(&[4, 3], 11);
    let mut a = mk::<i32, U2>(&da, &s);
    let b = mk::<i32, U2>(&db, &[4, 3]);
    a.zip_with_in_place(&b.t_view(), |x, y| x - y).unwrap();
    let (_, tdb) = ref_transpose(&[4, 3], &db);
    let expected: Vec<i32> = da.iter().zip(tdb.iter()).map(|(&x, &y)| x - y).collect();
    assert_arr("zip_with_in_place/strided_rhs", &a, &s, &expected);

    let s3 = [2, 3, 4];
    let dc = ints(&s3, 1);
    let dd = ints(&[2, 4], 7);
    let mut c = mk::<i32, U3>(&dc, &s3);
    let d = mk::<i32, U2>(&dd, &[2, 4]);
    {
        let mut sub = c.index_axis_mut(1, 1).unwrap();
        sub.zip_with_in_place(&d, |x, y| x - y).unwrap();
    }
    let mut expected2 = dc.clone();
    for i in 0..2 {
        for k in 0..4 {
            expected2[flat(&s3, &[i, 1, k])] -= dd[flat(&[2, 4], &[i, k])];
        }
    }
    assert_arr("zip_with_in_place/strided_receiver", &c, &s3, &expected2);
}

#[test]
fn scan_all_axes() {
    // scanr accumulates top-down, scanl bottom-up.
    let dg = ints(&[2, 3], 1);
    assert_eq!(dg, vec![1, 2, -3, 4, 5, -6]);
    let g = mk::<i32, U2>(&dg, &[2, 3]);
    assert_arr(
        "scan_golden/scanr0",
        &g.scanr(0, |x, y| x + y),
        &[2, 3],
        &[1, 2, -3, 5, 7, -9],
    );
    assert_arr(
        "scan_golden/scanr1",
        &g.scanr(1, |x, y| x + y),
        &[2, 3],
        &[1, 3, 0, 4, 9, 3],
    );
    assert_arr(
        "scan_golden/scanl0",
        &g.scanl(0, |x, y| x + y),
        &[2, 3],
        &[5, 7, -9, 4, 5, -6],
    );
    assert_arr(
        "scan_golden/scanl1",
        &g.scanl(1, |x, y| x + y),
        &[2, 3],
        &[0, -1, -3, 3, -1, -6],
    );

    for shape in [&[2, 3][..], &[3, 4][..], &[1, 5][..], &[5, 1][..]] {
        let d = ints(shape, 1);
        let a = mk::<i32, U2>(&d, shape);
        for axis in 0..2 {
            assert_arr(
                &format!("scanr({axis})/{shape:?}"),
                &a.scanr(axis, |x, y| x + y),
                shape,
                &ref_scanr(shape, &d, axis, |x, y| x + y),
            );
            assert_arr(
                &format!("scanl({axis})/{shape:?}"),
                &a.scanl(axis, |x, y| x + y),
                shape,
                &ref_scanl(shape, &d, axis, |x, y| x + y),
            );
            // Non-commutative operator pins the fold order.
            assert_arr(
                &format!("scanr_sub({axis})/{shape:?}"),
                &a.scanr(axis, |x, y| x - y),
                shape,
                &ref_scanr(shape, &d, axis, |x, y| x - y),
            );
            assert_arr(
                &format!("scanl_sub({axis})/{shape:?}"),
                &a.scanl(axis, |x, y| x - y),
                shape,
                &ref_scanl(shape, &d, axis, |x, y| x - y),
            );
        }
    }
}

#[test]
fn scan_rank3_and_rank1() {
    let s = [2, 3, 4];
    let d = ints(&s, 1);
    let a = mk::<i32, U3>(&d, &s);
    for axis in 0..3 {
        assert_arr(
            &format!("scanr3({axis})"),
            &a.scanr(axis, |x, y| x - y),
            &s,
            &ref_scanr(&s, &d, axis, |x, y| x - y),
        );
        assert_arr(
            &format!("scanl3({axis})"),
            &a.scanl(axis, |x, y| x - y),
            &s,
            &ref_scanl(&s, &d, axis, |x, y| x - y),
        );
    }

    // Golden rank-1 values: scanr(sub): out[k] = x[k] - out[k-1]; scanl(sub): out[k] = x[k] - out[k+1].
    let dv = ints(&[6], 1);
    assert_eq!(dv, vec![1, 2, -3, 4, 5, -6]);
    let v = mk::<i32, U1>(&dv, &[6]);
    assert_arr(
        "scanr1",
        &v.scanr(0, |x, y| x - y),
        &[6],
        &[1, 1, -4, 8, -3, -3],
    );
    assert_arr(
        "scanl1",
        &v.scanl(0, |x, y| x - y),
        &[6],
        &[3, -2, 4, -7, 11, -6],
    );

    let one = mk::<i32, U1>(&[42], &[1]);
    assert_arr("scanr1_single", &one.scanr(0, |x, y| x + y), &[1], &[42]);
    assert_arr("scanl1_single", &one.scanl(0, |x, y| x + y), &[1], &[42]);
}

#[test]
fn scan_on_transposed_input() {
    let s = [2, 3, 4];
    let d = ints(&s, 1);
    let a = mk::<i32, U3>(&d, &s);
    let t = a.t();
    let (ts, td) = ref_transpose(&s, &d);
    for axis in 0..3 {
        assert_arr(
            &format!("scanr_t({axis})"),
            &t.scanr(axis, |x, y| x - y),
            &ts,
            &ref_scanr(&ts, &td, axis, |x, y| x - y),
        );
    }
}

#[test]
fn scan_random() {
    let mut g = rng();
    for trial in 0..N_TRIALS {
        let d0 = g.gen_usize(1, 4);
        let d1 = g.gen_usize(1, 5);
        let shape = [d0, d1];
        let d = ints(&shape, trial as i32 + 1);
        let a = mk::<i32, U2>(&d, &shape);
        let axis = g.gen_usize(0, 1);
        assert_arr(
            &format!("rand_scanr[{trial}]"),
            &a.scanr(axis, |x, y| x - y),
            &shape,
            &ref_scanr(&shape, &d, axis, |x, y| x - y),
        );
        assert_arr(
            &format!("rand_scanl[{trial}]"),
            &a.scanl(axis, |x, y| x * 2 - y),
            &shape,
            &ref_scanl(&shape, &d, axis, |x, y| x * 2 - y),
        );
    }
}

#[test]
fn mat_mul_non_square() {
    for (sa, sb) in [
        (&[2, 3][..], &[3, 4][..]),
        (&[1, 5][..], &[5, 1][..]),
        (&[5, 1][..], &[1, 5][..]),
        (&[1, 1][..], &[1, 1][..]),
        (&[4, 2][..], &[2, 7][..]),
    ] {
        let da = ints(sa, 1);
        let db = ints(sb, 5);
        let a = mk::<i32, U2>(&da, sa);
        let b = mk::<i32, U2>(&db, sb);
        let (es, ed) = ref_matmul(sa, &da, sb, &db);
        assert_arr(
            &format!("mat_mul{sa:?}x{sb:?}"),
            &a.mat_mul(&b).unwrap(),
            &es,
            &ed,
        );
    }
}

#[test]
fn mat_mul_mixed_rank() {
    let dm = ints(&[3, 4], 1);
    let dv = ints(&[4], 2);
    let m = mk::<i32, U2>(&dm, &[3, 4]);
    let v = mk::<i32, U1>(&dv, &[4]);
    // Vectors are `inner_product` (tensordot) cases; `mat_mul` needs two axes.
    let dot =
        |x: &Ndarr<i32, U1>, y: &Ndarr<i32, U1>| x.inner_product(y, |a, b| a * b, |a, b| a + b);
    let (es, ed) = ref_matmul(&[3, 4], &dm, &[4], &dv);
    assert_arr(
        "inner/mat_vec",
        &m.inner_product(&v, |a, b| a * b, |a, b| a + b).unwrap(),
        &es,
        &ed,
    );

    let du = ints(&[3], 3);
    let u = mk::<i32, U1>(&du, &[3]);
    let (es, ed) = ref_matmul(&[3], &du, &[3, 4], &dm);
    assert_arr(
        "inner/vec_mat",
        &u.inner_product(&m, |a, b| a * b, |a, b| a + b).unwrap(),
        &es,
        &ed,
    );

    let dw = ints(&[6], 1);
    let w = mk::<i32, U1>(&dw, &[6]);
    let (es, ed) = ref_matmul(&[6], &dw, &[6], &dw);
    assert_arr("inner/dot", &dot(&w, &w).unwrap(), &es, &ed);
    assert_eq!(dot(&w, &w).unwrap().scalar(), ed[0]);

    let dt3 = ints(&[2, 3, 4], 1);
    let dn = ints(&[4, 5], 2);
    let t3 = mk::<i32, U3>(&dt3, &[2, 3, 4]);
    let n = mk::<i32, U2>(&dn, &[4, 5]);
    let (es, ed) = ref_matmul(&[2, 3, 4], &dt3, &[4, 5], &dn);
    assert_arr("mat_mul/rank3_rank2", &t3.mat_mul(&n).unwrap(), &es, &ed);

    let dp = ints(&[3, 2, 2], 1);
    let dq = ints(&[2, 2, 3], 1);
    let p = mk::<i32, U3>(&dp, &[3, 2, 2]);
    let q = mk::<i32, U3>(&dq, &[2, 2, 3]);
    let (es, ed) = ref_matmul(&[3, 2, 2], &dp, &[2, 2, 3], &dq);
    assert_arr(
        "inner/rank3_rank3",
        &p.inner_product(&q, |a, b| a * b, |a, b| a + b).unwrap(),
        &es,
        &ed,
    );
    // `mat_mul` batches leading axes instead, and batches 3 and 2 do not broadcast.
    assert!(p.mat_mul(&q).is_err());
}

#[test]
fn mat_mul_non_contiguous_inputs() {
    let da = ints(&[2, 3], 1);
    let db = ints(&[2, 3], 9);
    let a = mk::<i32, U2>(&da, &[2, 3]);
    let b = mk::<i32, U2>(&db, &[2, 3]);
    let (tas, tad) = ref_transpose(&[2, 3], &da);
    let (tbs, tbd) = ref_transpose(&[2, 3], &db);

    let (es, ed) = ref_matmul(&tas, &tad, &[2, 3], &db);
    assert_arr("mat_mul/t_lhs", &a.t().mat_mul(&b).unwrap(), &es, &ed);
    let (es, ed) = ref_matmul(&[2, 3], &da, &tbs, &tbd);
    assert_arr("mat_mul/t_rhs", &a.mat_mul(&b.t()).unwrap(), &es, &ed);
    let (es, ed) = ref_matmul(&tas, &tad, &[2, 3], &db);
    assert_arr(
        "mat_mul/t_both",
        &a.t().mat_mul(&b.t().t()).unwrap(),
        &es,
        &ed,
    );

    let dc = ints(&[2, 3, 4], 1);
    let c = mk::<i32, U3>(&dc, &[2, 3, 4]);
    let sl = c.slice_at(0)[1].clone();
    let (sls, sld) = ref_index_axis(&[2, 3, 4], &dc, 0, 1);
    let (tsls, tsld) = ref_transpose(&sls, &sld);
    let (es, ed) = ref_matmul(&sls, &sld, &tsls, &tsld);
    assert_arr(
        "mat_mul/sliced_lhs",
        &sl.mat_mul(&sl.t()).unwrap(),
        &es,
        &ed,
    );
}

#[test]
fn mat_mul_float() {
    let mut g = rng();
    let da = floats(&[3, 5], &mut g);
    let db = floats(&[5, 2], &mut g);
    let a = mk::<f64, U2>(&da, &[3, 5]);
    let b = mk::<f64, U2>(&db, &[5, 2]);
    let (es, ed) = ref_inner(&[3, 5], &da, &[5, 2], &db, |x, y| x * y, |x, y| x + y);
    assert_arr_approx("mat_mul/f64", &a.mat_mul(&b).unwrap(), &es, &ed, 1e-9);
}

#[test]
fn mat_mul_random() {
    let mut g = rng();
    for trial in 0..N_TRIALS {
        let m = g.gen_usize(1, 5);
        let k = g.gen_usize(1, 5);
        let n = g.gen_usize(1, 5);
        let da = ints(&[m, k], trial as i32 + 1);
        let db = ints(&[k, n], trial as i32 + 41);
        let a = mk::<i32, U2>(&da, &[m, k]);
        let b = mk::<i32, U2>(&db, &[k, n]);
        let (es, ed) = ref_matmul(&[m, k], &da, &[k, n], &db);
        assert_arr(
            &format!("rand_matmul[{trial}] {m}x{k}x{n}"),
            &a.mat_mul(&b).unwrap(),
            &es,
            &ed,
        );
    }
}

#[test]
fn inner_product_non_commutative() {
    let f = |x: i32, y: i32| x - 2 * y;
    let g = |x: i32, y: i32| x - y;
    for (sa, sb) in [
        (&[2, 3][..], &[3, 4][..]),
        (&[3, 4][..], &[4, 1][..]),
        (&[1, 6][..], &[6, 2][..]),
    ] {
        let da = ints(sa, 1);
        let db = ints(sb, 7);
        let a = mk::<i32, U2>(&da, sa);
        let b = mk::<i32, U2>(&db, sb);
        let (es, ed) = ref_inner(sa, &da, sb, &db, f, g);
        assert_arr(
            &format!("inner_nc{sa:?}x{sb:?}"),
            &a.inner_product(&b, |x, y| f(*x, *y), g).unwrap(),
            &es,
            &ed,
        );
    }

    let dv = ints(&[5], 1);
    let dw = ints(&[5], 9);
    let v = mk::<i32, U1>(&dv, &[5]);
    let w = mk::<i32, U1>(&dw, &[5]);
    let (es, ed) = ref_inner(&[5], &dv, &[5], &dw, f, g);
    assert_arr(
        "inner_nc/vec",
        &v.inner_product(&w, |x, y| f(*x, *y), g).unwrap(),
        &es,
        &ed,
    );
    let (es, ed) = ref_inner(&[5], &dw, &[5], &dv, f, g);
    assert_arr(
        "inner_nc/vec_rev",
        &w.inner_product(&v, |x, y| f(*x, *y), g).unwrap(),
        &es,
        &ed,
    );

    let dm = ints(&[3, 5], 2);
    let m = mk::<i32, U2>(&dm, &[3, 5]);
    let (es, ed) = ref_inner(&[3, 5], &dm, &[5], &dv, f, g);
    assert_arr(
        "inner_nc/mat_vec",
        &m.inner_product(&v, |x, y| f(*x, *y), g).unwrap(),
        &es,
        &ed,
    );
    let dm2 = ints(&[5, 3], 2);
    let m2 = mk::<i32, U2>(&dm2, &[5, 3]);
    let (es, ed) = ref_inner(&[5], &dv, &[5, 3], &dm2, f, g);
    assert_arr(
        "inner_nc/vec_mat",
        &v.inner_product(&m2, |x, y| f(*x, *y), g).unwrap(),
        &es,
        &ed,
    );
}

#[test]
fn inner_product_rank3_and_types() {
    let da = ints(&[2, 2, 3], 1);
    let db = ints(&[3, 2, 2], 1);
    let a = mk::<i32, U3>(&da, &[2, 2, 3]);
    let b = mk::<i32, U3>(&db, &[3, 2, 2]);
    let f = |x: i32, y: i32| x * y;
    let g = |x: i32, y: i32| x + y;
    let (es, ed) = ref_inner(&[2, 2, 3], &da, &[3, 2, 2], &db, f, g);
    assert_arr(
        "inner/rank3",
        &a.inner_product(&b, |x, y| f(*x, *y), g).unwrap(),
        &es,
        &ed,
    );
    // `mat_mul` batches leading axes rather than taking `tensordot`: the
    // [2] and [3] batches do not broadcast.
    assert!(a.mat_mul(&b).is_err());

    let hf = |x: i32, y: i32| (x + y) as f64 / 2.0;
    let hg = |x: f64, y: f64| x.max(y);
    let dc = ints(&[3, 4], 1);
    let dd = ints(&[4, 2], 3);
    let c = mk::<i32, U2>(&dc, &[3, 4]);
    let d = mk::<i32, U2>(&dd, &[4, 2]);
    let r = c.inner_product(&d, |x, y| hf(*x, *y), hg).unwrap();
    let (es, ed) = ref_inner(&[3, 4], &dc, &[4, 2], &dd, hf, hg);
    assert_arr_approx("inner/hetero", &r, &es, &ed, 1e-12);
}

#[test]
fn outer_product_asymmetric_both_orders() {
    let f = |x: i32, y: i32| x - y;
    for (sa, sb) in [
        (&[3][..], &[5][..]),
        (&[1][..], &[4][..]),
        (&[2, 3][..], &[4][..]),
        (&[2, 3][..], &[3, 2][..]),
        (&[1, 5][..], &[5, 1][..]),
    ] {
        let da = ints(sa, 1);
        let db = ints(sb, 6);
        let (fs, fd) = ref_outer(sa, &da, sb, &db, f);
        let (rs, rd) = ref_outer(sb, &db, sa, &da, f);
        match (sa.len(), sb.len()) {
            (1, 1) => {
                let a = mk::<i32, U1>(&da, sa);
                let b = mk::<i32, U1>(&db, sb);
                assert_arr(
                    &format!("outer{sa:?}x{sb:?}"),
                    &a.outer_product(&b, |x, y| f(*x, *y)).unwrap(),
                    &fs,
                    &fd,
                );
                assert_arr(
                    &format!("outer_rev{sa:?}x{sb:?}"),
                    &b.outer_product(&a, |x, y| f(*x, *y)).unwrap(),
                    &rs,
                    &rd,
                );
            }
            (2, 1) => {
                let a = mk::<i32, U2>(&da, sa);
                let b = mk::<i32, U1>(&db, sb);
                assert_arr(
                    &format!("outer{sa:?}x{sb:?}"),
                    &a.outer_product(&b, |x, y| f(*x, *y)).unwrap(),
                    &fs,
                    &fd,
                );
                assert_arr(
                    &format!("outer_rev{sa:?}x{sb:?}"),
                    &b.outer_product(&a, |x, y| f(*x, *y)).unwrap(),
                    &rs,
                    &rd,
                );
            }
            _ => {
                let a = mk::<i32, U2>(&da, sa);
                let b = mk::<i32, U2>(&db, sb);
                assert_arr(
                    &format!("outer{sa:?}x{sb:?}"),
                    &a.outer_product(&b, |x, y| f(*x, *y)).unwrap(),
                    &fs,
                    &fd,
                );
                assert_arr(
                    &format!("outer_rev{sa:?}x{sb:?}"),
                    &b.outer_product(&a, |x, y| f(*x, *y)).unwrap(),
                    &rs,
                    &rd,
                );
            }
        }
    }
}

#[test]
fn outer_product_rank0_and_views() {
    let f = |x: i32, y: i32| x * 10 + y;
    let s = mk::<i32, U0>(&[3], &[]);
    let dv = ints(&[4], 1);
    let v = mk::<i32, U1>(&dv, &[4]);
    let (es, ed) = ref_outer(&[], &[3], &[4], &dv, f);
    assert_arr(
        "outer/rank0_lhs",
        &s.outer_product(&v, |x, y| f(*x, *y)).unwrap(),
        &es,
        &ed,
    );
    let (es, ed) = ref_outer(&[4], &dv, &[], &[3], f);
    assert_arr(
        "outer/rank0_rhs",
        &v.outer_product(&s, |x, y| f(*x, *y)).unwrap(),
        &es,
        &ed,
    );

    let dm = ints(&[2, 3], 1);
    let m = mk::<i32, U2>(&dm, &[2, 3]);
    let (tms, tmd) = ref_transpose(&[2, 3], &dm);
    let (es, ed) = ref_outer(&tms, &tmd, &[4], &dv, f);
    assert_arr(
        "outer/transposed",
        &m.t().outer_product(&v, |x, y| f(*x, *y)).unwrap(),
        &es,
        &ed,
    );

    let rolled_m = ref_roll(&[2, 3], &dm, 1, 1);
    let rolled_v = ref_roll(&[4], &dv, -1, 0);
    let (es, ed) = ref_outer(&[2, 3], &rolled_m, &[4], &rolled_v, f);
    assert_arr(
        "outer/rolled",
        &m.roll(1, 1)
            .outer_product(&v.roll(-1, 0), |x, y| f(*x, *y))
            .unwrap(),
        &es,
        &ed,
    );
}

#[test]
fn outer_product_random() {
    let mut g = rng();
    let f = |x: i32, y: i32| x - 3 * y;
    for trial in 0..N_TRIALS {
        let n = g.gen_usize(1, 5);
        let m = g.gen_usize(1, 5);
        let da = ints(&[n], trial as i32 + 1);
        let db = ints(&[m], trial as i32 + 23);
        let a = mk::<i32, U1>(&da, &[n]);
        let b = mk::<i32, U1>(&db, &[m]);
        let (es, ed) = ref_outer(&[n], &da, &[m], &db, f);
        assert_arr(
            &format!("rand_outer[{trial}]"),
            &a.outer_product(&b, |x, y| f(*x, *y)).unwrap(),
            &es,
            &ed,
        );
        let (es, ed) = ref_outer(&[m], &db, &[n], &da, f);
        assert_arr(
            &format!("rand_outer_rev[{trial}]"),
            &b.outer_product(&a, |x, y| f(*x, *y)).unwrap(),
            &es,
            &ed,
        );
    }
}

#[test]
fn outer_product_rank4_result() {
    let f = |x: i32, y: i32| x + y;
    let da = ints(&[2, 3], 1);
    let db = ints(&[4, 2], 5);
    let a = mk::<i32, U2>(&da, &[2, 3]);
    let b = mk::<i32, U2>(&db, &[4, 2]);
    let r: Ndarr<i32, U4> = a.outer_product(&b, |x, y| f(*x, *y)).unwrap();
    let (es, ed) = ref_outer(&[2, 3], &da, &[4, 2], &db, f);
    assert_arr("outer/rank4", &r, &es, &ed);
    assert_eq!(r.shape(), &[2, 3, 4, 2]);
}

#[test]
fn float_unary_family() {
    let mut g = rng();
    let eps = 1e-12;
    for shape in [&[6][..], &[2, 3][..], &[1, 5][..], &[2, 3, 4][..]] {
        let d = floats(shape, &mut g);
        let a = mk_dyn(&d, shape);
        let fmap = |h: fn(f64) -> f64| d.iter().map(|&x| h(x)).collect::<Vec<f64>>();
        assert_arr_approx(
            &format!("sin{shape:?}"),
            &a.sin(),
            shape,
            &fmap(f64::sin),
            eps,
        );
        assert_arr_approx(
            &format!("cos{shape:?}"),
            &a.cos(),
            shape,
            &fmap(f64::cos),
            eps,
        );
        assert_arr_approx(
            &format!("tan{shape:?}"),
            &a.tan(),
            shape,
            &fmap(f64::tan),
            eps,
        );
        assert_arr_approx(
            &format!("sinh{shape:?}"),
            &a.sinh(),
            shape,
            &fmap(f64::sinh),
            eps,
        );
        assert_arr_approx(
            &format!("cosh{shape:?}"),
            &a.cosh(),
            shape,
            &fmap(f64::cosh),
            eps,
        );
        assert_arr_approx(
            &format!("tanh{shape:?}"),
            &a.tanh(),
            shape,
            &fmap(f64::tanh),
            eps,
        );
        assert_arr_approx(
            &format!("exp{shape:?}"),
            &a.exp(),
            shape,
            &fmap(f64::exp),
            eps,
        );
        assert_arr_approx(&format!("ln{shape:?}"), &a.ln(), shape, &fmap(f64::ln), eps);
        assert_arr_approx(
            &format!("log2{shape:?}"),
            &a.log2(),
            shape,
            &fmap(f64::log2),
            eps,
        );
        assert_arr_approx(
            &format!("log3{shape:?}"),
            &a.log(3.0),
            shape,
            &fmap(|x| x.log(3.0)),
            eps,
        );
        let bmap = |h: fn(f64) -> bool| d.iter().map(|&x| h(x)).collect::<Vec<bool>>();
        assert_arr(
            &format!("is_nan{shape:?}"),
            &a.is_nan(),
            shape,
            &bmap(f64::is_nan),
        );
        assert_arr(
            &format!("is_finite{shape:?}"),
            &a.is_finite(),
            shape,
            &bmap(f64::is_finite),
        );
        assert_arr(
            &format!("is_infinite{shape:?}"),
            &a.is_infinite(),
            shape,
            &bmap(f64::is_infinite),
        );
        assert_arr(
            &format!("is_normal{shape:?}"),
            &a.is_normal(),
            shape,
            &bmap(f64::is_normal),
        );
        let rmax = d.iter().copied().reduce(f64::max).unwrap();
        let rmin = d.iter().copied().reduce(f64::min).unwrap();
        let amax = a.iter_elems().cloned().reduce(f64::max).unwrap();
        let amin = a.iter_elems().cloned().reduce(f64::min).unwrap();
        assert!((amax - rmax).abs() < 1e-12, "maxf{shape:?}");
        assert!((amin - rmin).abs() < 1e-12, "minf{shape:?}");
    }
}

#[test]
fn float_special_values() {
    let d = vec![
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        0.0,
        -0.0,
        1e-320,
        -3.5,
        7.25,
    ];
    let a = mk::<f64, U1>(&d, &[8]);
    let bmap = |h: fn(f64) -> bool| d.iter().map(|&x| h(x)).collect::<Vec<bool>>();
    assert_arr("special/is_nan", &a.is_nan(), &[8], &bmap(f64::is_nan));
    assert_arr(
        "special/is_finite",
        &a.is_finite(),
        &[8],
        &bmap(f64::is_finite),
    );
    assert_arr(
        "special/is_infinite",
        &a.is_infinite(),
        &[8],
        &bmap(f64::is_infinite),
    );
    assert_arr(
        "special/is_normal",
        &a.is_normal(),
        &[8],
        &bmap(f64::is_normal),
    );
    let fmap = |h: fn(f64) -> f64| d.iter().map(|&x| h(x)).collect::<Vec<f64>>();
    assert_arr_approx("special/exp", &a.exp(), &[8], &fmap(f64::exp), 1e-12);
    assert_arr_approx("special/tanh", &a.tanh(), &[8], &fmap(f64::tanh), 1e-12);
    // IEEE max/min (NaN-ignoring) via the element iterator: golden values.
    assert_eq!(
        a.iter_elems().cloned().reduce(f64::max),
        Some(f64::INFINITY),
        "special/maxf"
    );
    assert_eq!(
        a.iter_elems().cloned().reduce(f64::min),
        Some(f64::NEG_INFINITY),
        "special/minf"
    );
}

#[test]
fn float_on_non_contiguous_source() {
    let mut g = rng();
    let s = [2, 3, 4];
    let d = floats(&s, &mut g);
    let a = mk::<f64, U3>(&d, &s);
    let eps = 1e-12;

    let (ts, td) = ref_transpose(&s, &d);
    let esin: Vec<f64> = td.iter().map(|x| x.sin()).collect();
    assert_arr_approx("nc/sin_t", &a.t().sin(), &ts, &esin, eps);

    let rolled = ref_roll(&s, &d, 2, 1);
    let eexp: Vec<f64> = rolled.iter().map(|x| x.exp()).collect();
    assert_arr_approx("nc/exp_roll", &a.roll(2, 1).exp(), &s, &eexp, eps);

    let sl = a.slice_at(2)[1].clone();
    let (sls, sld) = ref_index_axis(&s, &d, 2, 1);
    let etanh: Vec<f64> = sld.iter().map(|x| x.tanh()).collect();
    assert_arr_approx("nc/tanh_slice", &sl.tanh(), &sls, &etanh, eps);
}

#[test]
fn extras_signed_and_reductions() {
    for shape in [&[6][..], &[2, 3][..], &[1, 1][..], &[2, 3, 4][..]] {
        let d = ints(shape, -5);
        let a = mk_dyn(&d, shape);
        let eabs: Vec<i32> = d.iter().map(|&x| x.abs()).collect();
        assert_arr(&format!("abs{shape:?}"), &a.abs(), shape, &eabs);
        let epos: Vec<bool> = d.iter().map(|&x| x > 0).collect();
        assert_arr(
            &format!("is_positive{shape:?}"),
            &a.is_positive(),
            shape,
            &epos,
        );
        let eneg: Vec<bool> = d.iter().map(|&x| x < 0).collect();
        assert_arr(
            &format!("is_negative{shape:?}"),
            &a.is_negative(),
            shape,
            &eneg,
        );
        let esum: i32 = d.iter().sum();
        assert_eq!(a.iter_elems().sum::<i32>(), esum, "sum{shape:?}");
        let emax = *d.iter().max().unwrap();
        assert_eq!(a.iter_elems().cloned().max().unwrap(), emax, "max{shape:?}");
    }

    let mut g = rng();
    let f = floats(&[3, 4], &mut g);
    let fa = mk::<f64, U2>(&f, &[3, 4]);
    let eabs: Vec<f64> = f.iter().map(|&x| x.abs()).collect();
    assert_arr_approx("abs/f64", &fa.abs(), &[3, 4], &eabs, 1e-12);
    let esum: f64 = f.iter().sum();
    assert!((fa.iter_elems().sum::<f64>() - esum).abs() < 1e-9);
}

#[test]
fn extras_on_transposed() {
    let s = [2, 3, 4];
    let d = ints(&s, -3);
    let a = mk::<i32, U3>(&d, &s);
    let (ts, td) = ref_transpose(&s, &d);
    let eabs: Vec<i32> = td.iter().map(|&x| x.abs()).collect();
    assert_arr("abs/t", &a.t().abs(), &ts, &eabs);
    assert_eq!(a.t_view().iter_elems().sum::<i32>(), d.iter().sum::<i32>());
    assert_eq!(
        a.t_view().iter_elems().cloned().max().unwrap(),
        *d.iter().max().unwrap()
    );

    let sl = a.slice_at(1)[2].clone();
    let (sls, sld) = ref_index_axis(&s, &d, 1, 2);
    assert_eq!(sl.iter_elems().sum::<i32>(), sld.iter().sum::<i32>());
    let eslabs: Vec<i32> = sld.iter().map(|&x| x.abs()).collect();
    assert_arr("abs/slice", &sl.abs(), &sls, &eslabs);
}

#[test]
fn random_pipeline() {
    let mut g = rng();
    for trial in 0..N_TRIALS {
        let d0 = g.gen_usize(1, 3);
        let d1 = g.gen_usize(1, 4);
        let d2 = g.gen_usize(1, 4);
        let s = [d0, d1, d2];
        let sb = [d1, d2];
        let da = ints(&s, trial as i32 + 1);
        let db = ints(&sb, trial as i32 + 13);
        let a = mk::<i32, U3>(&da, &s);
        let b = mk::<i32, U2>(&db, &sb);

        let na = (&(&a.t().t() * 2) - &b) % &(&b + 3);
        let a2: Vec<i32> = da.iter().map(|&x| x * 2).collect();
        let (s1, sub) = ref_zip(&s, &a2, &sb, &db, |x, y| x - y);
        let b3: Vec<i32> = db.iter().map(|&x| x + 3).collect();
        let (s2, expected) = ref_zip(&s1, &sub, &sb, &b3, |x, y| x % y);
        assert_arr(&format!("pipeline[{trial}]"), &na, &s2, &expected);

        let axis = g.gen_usize(0, 2);
        let (rs, rd) = ref_reduce(&s2, &expected, axis, |x, y| x - y);
        assert_arr(
            &format!("pipeline_reduce[{trial}]"),
            &na.reduce(axis, |x, y| x - y).unwrap(),
            &rs,
            &rd,
        );

        let mut acc = a.clone();
        let bb = b.broadcast_view_to(&Dim::from(s)).unwrap().to_owned_array();
        acc += &bb;
        let (as_, ad) = ref_zip(&s, &da, &sb, &db, |x, y| x + y);
        assert_arr(&format!("pipeline_assign[{trial}]"), &acc, &as_, &ad);
    }
}

#[test]
fn mat_mul_degenerate_contraction() {
    // Contracting axes that only line up because of length-1 broadcasting.
    let f = |x: i32, y: i32| x - y;
    let g = |x: i32, y: i32| x * 2 - y;
    for (sa, sb) in [
        (&[2, 3][..], &[1, 4][..]),
        (&[2, 3][..], &[1, 1][..]),
        (&[2, 1][..], &[1, 3][..]),
        (&[1, 3][..], &[3, 1][..]),
        (&[2, 3][..], &[1, 3][..]),
        (&[4, 1][..], &[1, 1][..]),
    ] {
        let da = ints(sa, 1);
        let db = ints(sb, 5);
        let a = mk::<i32, U2>(&da, sa);
        let b = mk::<i32, U2>(&db, sb);
        let (es, ed) = ref_matmul(sa, &da, sb, &db);
        assert_arr(
            &format!("degen_matmul{sa:?}x{sb:?}"),
            &a.mat_mul(&b).unwrap(),
            &es,
            &ed,
        );
        let (es, ed) = ref_inner(sa, &da, sb, &db, f, g);
        assert_arr(
            &format!("degen_inner{sa:?}x{sb:?}"),
            &a.inner_product(&b, |x, y| f(*x, *y), g).unwrap(),
            &es,
            &ed,
        );
    }
}

#[test]
fn mat_mul_inner_random_mixed_rank() {
    let mut g = rng();
    let f = |x: i32, y: i32| x * y;
    let h = |x: i32, y: i32| x + y;
    for trial in 0..N_TRIALS {
        let d0 = g.gen_usize(1, 3);
        let k = g.gen_usize(1, 4);
        let n = g.gen_usize(1, 4);
        let sa = [d0, k, n];
        let sb = [n, k];
        let da = ints(&sa, trial as i32 + 1);
        let db = ints(&sb, trial as i32 + 17);
        let a = mk::<i32, U3>(&da, &sa);
        let b = mk::<i32, U2>(&db, &sb);
        let (es, ed) = ref_matmul(&sa, &da, &sb, &db);
        assert_arr(
            &format!("rand_mm_r3r2[{trial}]"),
            &a.mat_mul(&b).unwrap(),
            &es,
            &ed,
        );
        let (es, ed) = ref_inner(&sa, &da, &sb, &db, f, h);
        assert_arr(
            &format!("rand_inner_r3r2[{trial}]"),
            &a.inner_product(&b, |x, y| f(*x, *y), h).unwrap(),
            &es,
            &ed,
        );
        let dv = ints(&[n], trial as i32 + 5);
        let v = mk::<i32, U1>(&dv, &[n]);
        let (es, ed) = ref_matmul(&sa, &da, &[n], &dv);
        assert_arr(
            &format!("rand_inner_r3r1[{trial}]"),
            &a.inner_product(&v, |x, y| x * y, |x, y| x + y).unwrap(),
            &es,
            &ed,
        );
    }
}

#[test]
fn zero_axis_broadcast_stays_empty() {
    // Zero-length axis meets length-1: NumPy semantics, the zero wins and the result is empty.
    let z = mk::<i32, U2>(&[], &[0, 3]);
    let o = mk::<i32, U2>(&[1, 2, 3], &[1, 3]);
    let v = mk::<i32, U1>(&[1, 2, 3], &[3]);
    for empty in [
        z.zip_with(&o, |x, y| x + y).unwrap(),
        o.zip_with(&z, |x, y| x + y).unwrap(),
        z.zip_with(&v, |x, y| x + y).unwrap(),
    ] {
        assert_eq!(empty.shape(), &[0, 3]);
        assert!(empty.data().is_empty());
    }
    // Matching zero-length axes stay legal.
    let z2 = mk::<i32, U2>(&[], &[0, 3]);
    assert_arr(
        "zero_axis_matched",
        &z.zip_with(&z2, |x, y| x + y).unwrap(),
        &[0, 3],
        &[],
    );
}

#[test]
fn slice_and_stack() {
    let data = seq(&[2, 3, 4], 0);
    let a = mk::<i32, U3>(&data, &[2, 3, 4]);
    for axis in 0..3 {
        let sa = a.slice_at(axis);
        assert_eq!(sa.len(), [2, 3, 4][axis], "slice_at({axis}) count");
        for (i, xa) in sa.iter().enumerate() {
            let (es, ed) = ref_index_axis(&[2, 3, 4], &data, axis, i);
            assert_arr(&format!("slice_at({axis})[{i}]"), xa, &es, &ed);
        }
        for i in 0..[2, 3, 4][axis] {
            let v = a.index_axis_view(axis, i).unwrap();
            let (es, ed) = ref_index_axis(&[2, 3, 4], &data, axis, i);
            assert_arr(
                &format!("index_axis_view({axis})[{i}]"),
                &v.to_owned_array(),
                &es,
                &ed,
            );
        }
        let da = stack(&sa, axis);
        assert_arr(&format!("stack({axis})"), &da, &[2, 3, 4], &data);
    }
}

#[test]
fn roll() {
    let data = seq(&[2, 3, 4], 0);
    let a = mk::<i32, U3>(&data, &[2, 3, 4]);
    for axis in 0..3 {
        for shift in [-2_isize, -1, 0, 1, 2, 5] {
            let ra = a.roll(shift, axis);
            let expected = ref_roll(&[2, 3, 4], &data, shift, axis);
            assert_arr(&format!("roll({shift},{axis})"), &ra, &[2, 3, 4], &expected);
        }
    }
}

#[test]
fn indexing() {
    let mut a = mk::<i32, U2>(&seq(&[2, 3], 0), &[2, 3]);
    assert_eq!(a[[1, 2]], 5);
    a[[0, 1]] = 99;
    assert_arr("index_mut", &a, &[2, 3], &[0, 99, 2, 3, 4, 5]);

    let sa = a.slice(s![1, ..]).unwrap().to_owned_array();
    assert_arr("slice(s![1, ..])", &sa, &[3], &[3, 4, 5]);
}

#[test]
fn random_structural() {
    let mut g = rng();
    for trial in 0..N_TRIALS {
        let d0 = g.gen_usize(1, 3);
        let d1 = g.gen_usize(1, 4);
        let d2 = g.gen_usize(1, 4);
        let shape = [d0, d1, d2];
        let data = seq(&shape, trial as i32);
        let a = mk::<i32, U3>(&data, &shape);

        let (ts, td) = ref_transpose(&shape, &data);
        assert_arr(&format!("rand[{trial}].t"), &a.t(), &ts, &td);

        let axis = g.gen_usize(0, 2);
        let sa = a.slice_at(axis);
        for (i, xa) in sa.iter().enumerate() {
            let (es, ed) = ref_index_axis(&shape, &data, axis, i);
            assert_arr(&format!("rand[{trial}].slice({axis})[{i}]"), xa, &es, &ed);
        }

        let shift = g.gen_isize(-4, 4);
        let expected = ref_roll(&shape, &data, shift, axis);
        assert_arr(
            &format!("rand[{trial}].roll"),
            &a.roll(shift, axis),
            &shape,
            &expected,
        );

        let (rs, rd) = ref_reduce(&shape, &data, axis, |x, y| x + y);
        assert_arr(
            &format!("rand[{trial}].reduce"),
            &a.reduce(axis, |x, y| x + y).unwrap(),
            &rs,
            &rd,
        );
    }
}

#[test]
fn random_broadcast_arith() {
    let mut g = rng();
    for trial in 0..N_TRIALS {
        let rows = g.gen_usize(1, 4);
        let cols = g.gen_usize(1, 5);
        let d1 = seq(&[rows, cols], 1);
        let d2 = seq(&[cols], 2);
        let a = mk::<i32, U2>(&d1, &[rows, cols]);
        let c = mk::<i32, U1>(&d2, &[cols]);

        let (s, d) = ref_zip(&[rows, cols], &d1, &[cols], &d2, |x, y| x + y);
        assert_arr(&format!("rand_badd[{trial}]"), &(&a + &c), &s, &d);
        let (s, d) = ref_zip(&[rows, cols], &d1, &[cols], &d2, |x, y| x * y);
        assert_arr(&format!("rand_bmul[{trial}]"), &(&a * &c), &s, &d);

        let dcol = seq(&[rows, 1], 3);
        let col = mk::<i32, U2>(&dcol, &[rows, 1]);
        let (s, d) = ref_zip(&[rows, cols], &d1, &[rows, 1], &dcol, |x, y| x * y);
        assert_arr(&format!("rand_col_bmul[{trial}]"), &(&a * &col), &s, &d);
    }
}
