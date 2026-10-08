//! Views, layout and broadcasting, checked against golden values and independent stride math.

use rapl::*;
use typenum::{U16, U32, U64};

/// Minimal 64-bit LCG (Knuth MMIX constants); high bits are used as output.
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Lcg(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0 >> 11
    }

    /// Uniform in `lo..=hi`.
    fn usize_in(&mut self, lo: usize, hi: usize) -> usize {
        lo + (self.next_u64() as usize) % (hi - lo + 1)
    }

    /// Uniform in `lo..hi`.
    fn i32_in(&mut self, lo: i32, hi: i32) -> i32 {
        lo + (self.next_u64() % ((hi - lo) as u64)) as i32
    }

    /// Uniform in `[lo, hi)`.
    fn f64_in(&mut self, lo: f64, hi: f64) -> f64 {
        let unit = (self.next_u64() % (1u64 << 53)) as f64 / (1u64 << 53) as f64;
        lo + unit * (hi - lo)
    }
}

const SEED: u64 = 0x000A_D17C_0DE5;
const N_RANDOM: usize = 96;

fn product(shape: &[usize]) -> usize {
    shape.iter().product()
}

fn assert_eq_slices<T: PartialEq + std::fmt::Debug>(
    op: &str,
    shape: &[usize],
    new: &[T],
    expected: &[T],
) {
    if new.len() != expected.len() {
        panic!(
            "{op}: length mismatch shape={shape:?} new_len={} expected_len={}",
            new.len(),
            expected.len()
        );
    }
    for (i, (a, b)) in new.iter().zip(expected.iter()).enumerate() {
        if a != b {
            panic!("{op}: mismatch at flat index {i} shape={shape:?}: new={a:?} expected={b:?}");
        }
    }
}

fn make_i32(shape: &[usize], start: i32) -> Vec<i32> {
    let n = product(shape);
    (0..n as i32).map(|i| start + i).collect()
}

fn a_u1(data: &[i32]) -> Ndarr<i32, U1> {
    Ndarr::new(data, [data.len()]).unwrap()
}

fn a_u2(data: &[i32], shape: [usize; 2]) -> Ndarr<i32, U2> {
    Ndarr::new(data, shape).unwrap()
}

fn a_u3(data: &[i32], shape: [usize; 3]) -> Ndarr<i32, U3> {
    Ndarr::new(data, shape).unwrap()
}

fn c_strides(shape: &[usize]) -> Vec<isize> {
    let r = shape.len();
    let mut strides = vec![1isize; r];
    for i in (0..r.saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1] as isize;
    }
    strides
}

/// Every multi-index of `shape` in C-order.
fn indices(shape: &[usize]) -> Vec<Vec<usize>> {
    if shape.contains(&0) {
        return Vec::new();
    }
    let mut out = vec![vec![0usize; shape.len()]];
    if shape.is_empty() {
        return out;
    }
    out.clear();
    let n = product(shape);
    let mut idx = vec![0usize; shape.len()];
    for _ in 0..n {
        out.push(idx.clone());
        for ax in (0..shape.len()).rev() {
            idx[ax] += 1;
            if idx[ax] < shape[ax] {
                break;
            }
            idx[ax] = 0;
        }
    }
    out
}

/// Materialize a strided window over `data` using only plain arithmetic.
fn gather<T: Copy>(data: &[T], offset: usize, strides: &[isize], shape: &[usize]) -> Vec<T> {
    indices(shape)
        .iter()
        .map(|ix| {
            let mut pos = offset as isize;
            for (k, &s) in strides.iter().enumerate() {
                pos += ix[k] as isize * s;
            }
            data[pos as usize]
        })
        .collect()
}

/// (offset, strides, shape) of fixing `axis` at `i` in a C-order array, from first principles.
fn slice_layout(shape: &[usize], axis: usize, i: usize) -> (usize, Vec<isize>, Vec<usize>) {
    let cs = c_strides(shape);
    let offset = (i as isize * cs[axis]) as usize;
    let mut strides = cs;
    strides.remove(axis);
    let mut sub = shape.to_vec();
    sub.remove(axis);
    (offset, strides, sub)
}

/// Expected transpose of a C-order array, from independent stride math.
fn transpose_expected(data: &[i32], shape: &[usize]) -> (Vec<i32>, Vec<usize>) {
    let mut rev_shape = shape.to_vec();
    rev_shape.reverse();
    let mut rev_strides = c_strides(shape);
    rev_strides.reverse();
    (gather(data, 0, &rev_strides, &rev_shape), rev_shape)
}

/// Left-fold reduction along `axis` of a C-order array.
fn reduce_expected<F: Fn(i32, i32) -> i32>(
    data: &[i32],
    shape: &[usize],
    axis: usize,
    f: F,
) -> Vec<i32> {
    let cs = c_strides(shape);
    let mut out_shape = shape.to_vec();
    out_shape.remove(axis);
    let mut out_strides = cs.clone();
    out_strides.remove(axis);
    indices(&out_shape)
        .iter()
        .map(|ix| {
            let base: isize = ix
                .iter()
                .zip(&out_strides)
                .map(|(&i, &s)| i as isize * s)
                .sum();
            let mut acc = data[base as usize];
            for j in 1..shape[axis] {
                acc = f(acc, data[(base + j as isize * cs[axis]) as usize]);
            }
            acc
        })
        .collect()
}

/// Roll along `axis`: output at index i reads input at (i - shift) mod len.
fn roll_expected(data: &[i32], shape: &[usize], shift: isize, axis: usize) -> Vec<i32> {
    let cs = c_strides(shape);
    let len = shape[axis] as isize;
    indices(shape)
        .iter()
        .map(|ix| {
            let mut src = ix.clone();
            src[axis] = (ix[axis] as isize - shift).rem_euclid(len) as usize;
            let pos: isize = src.iter().zip(&cs).map(|(&i, &s)| i as isize * s).sum();
            data[pos as usize]
        })
        .collect()
}

/// View/array invariants plus accessor vs. materialized agreement; a macro because the storage trait is crate-private.
macro_rules! assert_view_invariants {
    ($op:expr, $v:expr) => {{
        let op = $op;
        let v = $v;
        let owned = v.to_owned_array();
        assert_eq!(
            owned.shape(),
            v.shape(),
            "{op}: shape changed on materialize"
        );
        assert_eq!(owned.offset(), 0, "{op}: owned array must have offset 0");
        assert_eq!(
            owned.strides(),
            c_strides(v.shape()).as_slice(),
            "{op}: owned array must have C-order strides, shape={:?}",
            v.shape()
        );
        assert_eq!(v.len(), product(v.shape()), "{op}: len != product(shape)");
        assert_eq!(v.rank(), v.shape().len(), "{op}: rank != shape len");
        assert_eq!(v.strides().len(), v.rank(), "{op}: stride count != rank");

        let iterated: Vec<_> = v.iter_elems().cloned().collect();
        assert_eq_slices(
            &format!("{op}/iter_elems"),
            v.shape(),
            &iterated,
            owned.data(),
        );

        if !v.is_empty() {
            for ix in indices(v.shape()) {
                assert_eq!(
                    v.flat_index(&ix).unwrap(),
                    {
                        let mut pos = v.offset() as isize;
                        for (k, &s) in v.strides().iter().enumerate() {
                            pos += ix[k] as isize * s;
                        }
                        pos as usize
                    },
                    "{op}: flat_index({ix:?}) mismatch"
                );
            }
        }
    }};
}

const SHAPES_3: [[usize; 3]; 6] = [
    [2, 3, 4],
    [3, 1, 4],
    [1, 1, 5],
    [4, 2, 1],
    [1, 1, 1],
    [2, 1, 3],
];
const SHAPES_2: [[usize; 2]; 6] = [[2, 3], [1, 5], [5, 1], [1, 1], [3, 3], [4, 2]];

#[test]
fn owned_arrays_are_contiguous_with_zero_offset() {
    for shape in SHAPES_3 {
        let data = make_i32(&shape, 0);
        let a = a_u3(&data, shape);
        assert_eq!(a.offset(), 0, "shape={shape:?}");
        assert_eq!(a.strides(), c_strides(&shape).as_slice(), "shape={shape:?}");
        assert!(a.is_standard_layout(), "shape={shape:?}");
        assert_eq!(a.len(), product(&shape), "shape={shape:?}");
        assert_eq!(a.shape(), &shape, "shape={shape:?}");
        assert!(!a.is_empty(), "shape={shape:?}");
        assert_eq_slices("owned_data", &shape, a.data(), &data);
        assert_view_invariants!("owned", &a);
    }
    for shape in SHAPES_2 {
        let data = make_i32(&shape, -7);
        let a = a_u2(&data, shape);
        assert_eq!(a.strides(), c_strides(&shape).as_slice(), "shape={shape:?}");
        assert_eq!(a.len(), product(&shape), "shape={shape:?}");
        assert_view_invariants!("owned2", &a);
    }
}

#[test]
fn rank_zero_and_single_element() {
    let dim = Dim::<U0>::new(&[]).unwrap();
    let a = Ndarr::new(&[42i32], dim).unwrap();

    assert_eq!(a.rank(), 0);
    assert_eq!(a.len(), 1);
    assert!(!a.is_empty());
    assert_eq!(a.strides(), &[] as &[isize]);
    assert!(a.is_standard_layout());
    assert_eq!(a[[]], 42);
    assert_eq!(a.flat_index(&[]).unwrap(), 0);
    assert_eq!(a.iter_elems().cloned().collect::<Vec<_>>(), vec![42]);
    assert_view_invariants!("rank0", &a);
    assert_eq_slices("rank0_t", &[], a.t().data(), &[42]);
    assert_eq_slices("rank0_view", &[], a.view().to_owned_array().data(), &[42]);
    assert_eq_slices(
        "rank0_t_view",
        &[],
        a.t_view().to_owned_array().data(),
        &[42],
    );

    let s = a_u1(&[5]);
    assert_eq!(s.strides(), &[1isize]);
    assert_view_invariants!("rank1_single", &s);
    assert_eq_slices("rank1_single_t", &[1], s.t().data(), &[5]);
    let sub = s.index_axis_view(0, 0).unwrap();
    assert_eq!(sub.rank(), 0);
    assert_eq!(sub[[]], 5);
    assert_view_invariants!("rank1_single_axis", &sub);
}

#[test]
fn flat_index_error_paths() {
    let a = a_u3(&make_i32(&[2, 3, 4], 0), [2, 3, 4]);
    assert!(a.flat_index(&[0, 0]).is_err(), "short index rank");
    assert!(a.flat_index(&[0, 0, 0, 0]).is_err(), "long index rank");
    assert!(a.flat_index(&[]).is_err(), "empty index rank");
    assert!(a.flat_index(&[2, 0, 0]).is_err(), "axis 0 out of bounds");
    assert!(a.flat_index(&[0, 3, 0]).is_err(), "axis 1 out of bounds");
    assert!(a.flat_index(&[0, 0, 4]).is_err(), "axis 2 out of bounds");
    assert!(a.flat_index(&[1, 2, 3]).is_ok(), "last valid index");

    // Views inherit the same checks against their own logical shape.
    let v = a.index_axis_view(1, 2).unwrap();
    assert_eq!(v.shape(), &[2, 4]);
    assert!(v.flat_index(&[0, 0, 0]).is_err());
    assert!(v.flat_index(&[2, 0]).is_err());
    assert!(v.flat_index(&[0, 4]).is_err());
    assert!(v.flat_index(&[1, 3]).is_ok());

    // A degenerate axis makes every index out of bounds.
    let z: Ndarr<i32, U2> = Ndarr::new(&[], [3usize, 0]).unwrap();
    assert!(z.flat_index(&[0, 0]).is_err());
    assert!(z.flat_index(&[2, 0]).is_err());
}

#[test]
fn index_axis_view_error_paths() {
    let a = a_u3(&make_i32(&[2, 3, 4], 0), [2, 3, 4]);
    assert!(a.index_axis_view(3, 0).is_err(), "axis == rank");
    assert!(a.index_axis_view(99, 0).is_err(), "axis >> rank");
    assert!(a.index_axis_view(0, 2).is_err(), "index == axis len");
    assert!(a.index_axis_view(2, 4).is_err(), "index == axis len (last)");
    assert!(a.index_axis_view(1, 3).is_err());

    let z: Ndarr<i32, U2> = Ndarr::new(&[], [3usize, 0]).unwrap();
    assert!(z.index_axis_view(1, 0).is_err(), "index into empty axis");
    assert!(
        z.index_axis_view(0, 0).is_ok(),
        "empty result is still a view"
    );

    let mut m = a.clone();
    assert!(m.index_axis_mut(3, 0).is_err());
    assert!(m.index_axis_mut(0, 5).is_err());
    assert!(m.index_axis_mut(2, 3).is_ok());
}

#[test]
fn broadcast_view_error_paths() {
    let row = a_u1(&make_i32(&[4], 10));
    let small = Dim::<U1>::new(&[4]).unwrap();
    assert!(row.broadcast_view_to(&small).is_ok());
    let mat = a_u2(&make_i32(&[2, 3], 0), [2, 3]);
    assert!(
        mat.broadcast_view_to(&Dim::<U1>::new(&[3]).unwrap())
            .is_err(),
        "target rank < input rank"
    );
    assert!(mat
        .broadcast_view_to(&Dim::<U2>::new(&[4, 3]).unwrap())
        .is_err());
    assert!(mat
        .broadcast_view_to(&Dim::<U3>::new(&[5, 2, 5]).unwrap())
        .is_err());
    assert!(row
        .broadcast_view_to(&Dim::<U2>::new(&[3, 5]).unwrap())
        .is_err());
    assert!(mat
        .broadcast_view_to(&Dim::<U2>::new(&[2, 1]).unwrap())
        .is_err());
}

#[test]
fn index_axis_view_matches_owned_slice() {
    for shape in SHAPES_3 {
        let data = make_i32(&shape, 3);
        let a = a_u3(&data, shape);
        for axis in 0..3 {
            for i in 0..shape[axis] {
                let v = a.index_axis_view(axis, i).unwrap();
                let tag = format!("slice[{shape:?}/{axis}/{i}]");
                let (offset, strides, sub_shape) = slice_layout(&shape, axis, i);
                assert_eq!(v.shape(), sub_shape.as_slice(), "{tag}: shape");
                assert_eq!(v.offset(), offset, "{tag}: offset");
                assert_eq!(v.strides(), strides.as_slice(), "{tag}: strides");
                assert_view_invariants!(&tag, &v);
                let expected = gather(&data, offset, &strides, &sub_shape);
                assert_eq_slices(&tag, v.shape(), v.to_owned_array().data(), &expected);
            }
        }
    }
}

#[test]
fn t_view_matches_owned_transpose() {
    for shape in SHAPES_3 {
        let data = make_i32(&shape, 1);
        let a = a_u3(&data, shape);
        let tv = a.t_view();
        let (expected, rev) = transpose_expected(&data, &shape);
        assert_eq!(tv.shape(), rev.as_slice(), "shape={shape:?}");
        let mut rev_strides = c_strides(&shape);
        rev_strides.reverse();
        assert_eq!(tv.strides(), rev_strides.as_slice(), "shape={shape:?}");
        assert_view_invariants!(&format!("t_view{shape:?}"), &tv);
        assert_eq_slices("t_view", &rev, tv.to_owned_array().data(), &expected);
        assert_eq_slices("t", &rev, a.t().data(), &expected);
        assert_eq_slices(
            "t_view_twice",
            &shape,
            tv.t_view().to_owned_array().data(),
            &data,
        );
    }
    for shape in SHAPES_2 {
        let data = make_i32(&shape, 100);
        let a = a_u2(&data, shape);
        let (expected, rev) = transpose_expected(&data, &shape);
        assert_eq_slices("t2", &rev, a.t().data(), &expected);
        assert_view_invariants!(&format!("t_view2{shape:?}"), &a.t_view());
    }
}

#[test]
fn broadcast_view_zero_strides_and_values() {
    let row_data = make_i32(&[4], 10);
    let row = a_u1(&row_data);
    for target in [[2usize, 3, 4], [1, 1, 4], [5, 1, 4]] {
        let dim = Dim::<U3>::new(&target).unwrap();
        let v = row.broadcast_view_to(&dim).unwrap();
        assert_eq!(v.shape(), &target);
        assert_eq!(
            v.strides(),
            &[0isize, 0, 1],
            "broadcast must use 0-strides on expanded axes, target={target:?}"
        );
        assert_view_invariants!(&format!("bcast{target:?}"), &v);
        let expected: Vec<i32> = (0..target[0] * target[1])
            .flat_map(|_| row_data.iter().copied())
            .collect();
        assert_eq_slices("bcast", &target, v.to_owned_array().data(), &expected);
    }

    let col_data = make_i32(&[3, 1], 7);
    let col = a_u2(&col_data, [3, 1]);
    let dim = Dim::<U2>::new(&[3, 5]).unwrap();
    let v = col.broadcast_view_to(&dim).unwrap();
    assert_eq!(v.strides(), &[1isize, 0]);
    assert_view_invariants!("bcast_col", &v);
    let expected: Vec<i32> = col_data
        .iter()
        .flat_map(|&x| std::iter::repeat_n(x, 5))
        .collect();
    assert_eq_slices("bcast_col", &[3, 5], v.to_owned_array().data(), &expected);
}

#[test]
fn broadcast_of_a_view_matches_owned_composition() {
    let shape = [2usize, 3, 4];
    let data = make_i32(&shape, 0);
    let a = a_u3(&data, shape);
    let target = [2usize, 3, 4];
    let dim = Dim::<U3>::new(&target).unwrap();

    for i in 0..2 {
        let sub = a.index_axis_view(0, i).unwrap();
        let v = sub.broadcast_view_to(&dim).unwrap();
        assert_view_invariants!(&format!("bcast_of_slice[{i}]"), &v);
        let block = &data[i * 12..(i + 1) * 12];
        let expected: Vec<i32> = (0..2).flat_map(|_| block.iter().copied()).collect();
        assert_eq_slices(
            "bcast_of_slice",
            &target,
            v.to_owned_array().data(),
            &expected,
        );
    }

    for i in 0..4 {
        let sub = a.index_axis_view(2, i).unwrap();
        assert_eq!(sub.strides(), &[12isize, 4]);
        let big = Dim::<U3>::new(&[5, 2, 3]).unwrap();
        let v = sub.broadcast_view_to(&big).unwrap();
        assert_eq!(v.strides(), &[0isize, 12, 4]);
        assert_view_invariants!(&format!("bcast_of_strided[{i}]"), &v);
        let (offset, strides, sub_shape) = slice_layout(&shape, 2, i);
        let slice_vals = gather(&data, offset, &strides, &sub_shape);
        let expected: Vec<i32> = (0..5).flat_map(|_| slice_vals.iter().copied()).collect();
        assert_eq_slices(
            "bcast_of_strided",
            &[5, 2, 3],
            v.to_owned_array().data(),
            &expected,
        );
    }
}

#[test]
fn transpose_of_broadcast_of_slice_matches_reference() {
    let shape = [2usize, 3, 4];
    let data = make_i32(&shape, 5);
    let a = a_u3(&data, shape);

    let mut checked = 0;
    for (axis, extent) in shape.iter().enumerate() {
        for i in 0..*extent {
            let sub = a.index_axis_view(axis, i).unwrap();
            // prepend a fresh leading axis so every subview shape is broadcastable
            let mut target = vec![5usize];
            target.extend_from_slice(sub.shape());
            let dim = Dim::<U3>::new(&target).unwrap();

            let v = sub.broadcast_view_to(&dim).unwrap();
            let chained = v.t_view();
            let mut rev = target.clone();
            rev.reverse();
            let tag = format!("chain[{axis}/{i}] target={target:?}");
            assert_eq!(chained.shape(), rev.as_slice(), "{tag}: shape");
            assert_eq!(
                chained.strides().last(),
                Some(&0isize),
                "{tag}: 0-stride must survive the transpose"
            );
            assert_view_invariants!(&tag, &chained);
            let (offset, sub_strides, _) = slice_layout(&shape, axis, i);
            let mut bcast_strides = vec![0isize];
            bcast_strides.extend_from_slice(&sub_strides);
            bcast_strides.reverse();
            let expected = gather(&data, offset, &bcast_strides, &rev);
            assert_eq_slices(&tag, &rev, chained.to_owned_array().data(), &expected);
            checked += 1;
        }
    }
    assert_eq!(checked, 9, "every axis/index combination must be exercised");
}

#[test]
fn rank_four_views_match_reference() {
    for shape in [[2usize, 1, 3, 2], [1, 2, 2, 3], [3, 2, 1, 1]] {
        let data = make_i32(&shape, 11);
        let a = Ndarr::new(&data, shape).unwrap();
        assert_eq!(a.strides(), c_strides(&shape).as_slice(), "shape={shape:?}");
        assert_view_invariants!(&format!("r4{shape:?}"), &a);

        let (t_expected, rev) = transpose_expected(&data, &shape);
        assert_view_invariants!(&format!("r4_t{shape:?}"), &a.t_view());
        assert_eq_slices(
            "r4_t",
            &rev,
            a.t_view().to_owned_array().data(),
            &t_expected,
        );

        for axis in 0..4 {
            for i in 0..shape[axis] {
                let v = a.index_axis_view(axis, i).unwrap();
                let tag = format!("r4_slice[{shape:?}/{axis}/{i}]");
                assert_view_invariants!(&tag, &v);
                let (offset, strides, sub_shape) = slice_layout(&shape, axis, i);
                let expected = gather(&data, offset, &strides, &sub_shape);
                assert_eq_slices(&tag, v.shape(), v.to_owned_array().data(), &expected);
                assert_view_invariants!(&format!("{tag}/t"), &v.t_view());
                let mut rev_strides = strides.clone();
                rev_strides.reverse();
                let mut rev_shape = sub_shape.clone();
                rev_shape.reverse();
                let t_expected = gather(&data, offset, &rev_strides, &rev_shape);
                assert_eq_slices(
                    &format!("{tag}/t"),
                    &rev_shape,
                    v.t_view().to_owned_array().data(),
                    &t_expected,
                );
            }
        }
    }
}

#[test]
fn zip_with_between_differently_strided_views() {
    let shape = [2usize, 3, 4];
    let d1 = make_i32(&shape, 0);
    let d2 = make_i32(&shape, 100);
    let a = a_u3(&d1, shape);
    let c = a_u3(&d2, shape);

    let av = a.view();
    let ct = c.t_view();
    let ctt = ct.t_view();
    let summed = av.zip_with(&ctt, |x, y| x + y).unwrap();
    let expected: Vec<i32> = d1.iter().zip(d2.iter()).map(|(x, y)| x + y).collect();
    assert_eq_slices("zip_with_views", &shape, summed.data(), &expected);

    // strided ⊕ 0-stride broadcast, both of shape [2,3]
    let sub = a.index_axis_view(2, 1).unwrap();
    let row_data = make_i32(&[3], 1);
    let row = a_u1(&row_data);
    let dim = Dim::<U2>::new(&[2, 3]).unwrap();
    let bcast = row.broadcast_view_to(&dim).unwrap();
    assert_eq!(bcast.strides(), &[0isize, 1]);
    let mixed = sub.zip_with(&bcast, |x, y| x * y).unwrap();
    let (offset, strides, sub_shape) = slice_layout(&shape, 2, 1);
    let sub_vals = gather(&d1, offset, &strides, &sub_shape);
    let bcast_vals: Vec<i32> = (0..2).flat_map(|_| row_data.iter().copied()).collect();
    let expected_mixed: Vec<i32> = sub_vals
        .iter()
        .zip(bcast_vals.iter())
        .map(|(x, y)| x * y)
        .collect();
    assert_eq_slices("zip_with_mixed", &[2, 3], mixed.data(), &expected_mixed);
    assert_eq!(mixed.offset(), 0);
    assert_eq!(mixed.strides(), c_strides(&[2, 3]).as_slice());

    // cross-check through the public eager broadcast path
    let eager = sub.zip_with(&row, |x, y| x * y).unwrap();
    assert_eq_slices(
        "zip_with_mixed_vs_eager",
        &[2, 3],
        mixed.data(),
        eager.data(),
    );
}

#[test]
fn nested_axis_views_match_nested_owned_slices() {
    for shape in SHAPES_3 {
        let data = make_i32(&shape, 2);
        let a = a_u3(&data, shape);
        for ax0 in 0..3 {
            for i in 0..shape[ax0] {
                let outer = a.index_axis_view(ax0, i).unwrap();
                let (o_offset, o_strides, o_shape) = slice_layout(&shape, ax0, i);
                for ax1 in 0..2 {
                    for j in 0..outer.shape()[ax1] {
                        let inner = outer.index_axis_view(ax1, j).unwrap();
                        let tag = format!("nested[{shape:?}/{ax0}/{i}/{ax1}/{j}]");
                        let i_offset = (o_offset as isize + j as isize * o_strides[ax1]) as usize;
                        let mut i_strides = o_strides.clone();
                        i_strides.remove(ax1);
                        let mut i_shape = o_shape.clone();
                        i_shape.remove(ax1);
                        assert_eq!(inner.shape(), i_shape.as_slice(), "{tag}: shape");
                        assert_eq!(inner.offset(), i_offset, "{tag}: offset");
                        assert_eq!(inner.strides(), i_strides.as_slice(), "{tag}: strides");
                        assert_view_invariants!(&tag, &inner);
                        let expected = gather(&data, i_offset, &i_strides, &i_shape);
                        assert_eq_slices(
                            &tag,
                            inner.shape(),
                            inner.to_owned_array().data(),
                            &expected,
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn map_agrees_on_contiguous_fast_path_and_strided_path() {
    let shape = [2usize, 3, 4];
    let data = make_i32(&shape, 1);
    let a = a_u3(&data, shape);
    for (axis, extent) in shape.iter().enumerate() {
        for i in 0..*extent {
            let v = a.index_axis_view(axis, i).unwrap();
            let mapped = v.map(|x| x * 3 - 1);
            let (offset, strides, sub_shape) = slice_layout(&shape, axis, i);
            let expected: Vec<i32> = gather(&data, offset, &strides, &sub_shape)
                .iter()
                .map(|x| x * 3 - 1)
                .collect();
            assert_eq_slices(
                &format!("map_view[{axis}/{i}]"),
                v.shape(),
                mapped.data(),
                &expected,
            );
            assert_eq!(mapped.offset(), 0);
            assert_eq!(mapped.strides(), c_strides(v.shape()).as_slice());
        }
    }
    let row_data = make_i32(&[4], 10);
    let row = a_u1(&row_data);
    let dim = Dim::<U3>::new(&[2, 3, 4]).unwrap();
    let mapped = row.broadcast_view_to(&dim).unwrap().map(|x| x + 1);
    let expected: Vec<i32> = (0..6)
        .flat_map(|_| row_data.iter().map(|x| x + 1))
        .collect();
    assert_eq_slices("map_bcast", &[2, 3, 4], mapped.data(), &expected);
    let mapped_t = a.t_view().map(|x| -x);
    let (t_vals, _) = transpose_expected(&data, &shape);
    let expected_t: Vec<i32> = t_vals.iter().map(|x| -x).collect();
    assert_eq_slices("map_t_view", mapped_t.shape(), mapped_t.data(), &expected_t);
}

#[test]
fn view_mut_writes_to_the_right_buffer_positions() {
    let shape = [2usize, 3];
    let data = make_i32(&shape, 0);

    // column 2 of a [2,3] array is buffer positions 2 and 5
    let mut a = Ndarr::new(&data, shape).unwrap();
    {
        let mut col = a.index_axis_mut(1, 2).unwrap();
        assert_eq!(col.shape(), &[2]);
        assert_eq!(col.strides(), &[3isize]);
        assert_eq!(col.offset(), 2);
        col[[0]] = -1;
        col[[1]] = -2;
    }
    assert_eq!(a.data(), &[0, 1, -1, 3, 4, -2]);

    {
        let mut row = a.index_axis_mut(0, 1).unwrap();
        row[[0]] = 30;
        assert!(row.flat_index(&[3]).is_err());
        assert!(row.flat_index(&[0, 0]).is_err());
    }
    assert_eq!(a.data(), &[0, 1, -1, 30, 4, -2]);

    {
        let mut whole = a.view_mut();
        assert_eq!(whole.offset(), 0);
        assert_eq!(whole.strides(), c_strides(&shape).as_slice());
        whole[[1, 2]] = 99;
    }
    assert_eq!(a.data(), &[0, 1, -1, 30, 4, 99]);

    a.data_mut()[0] = 7;
    assert_eq!(a.data(), &[7, 1, -1, 30, 4, 99]);

    // map_in_place through a strided view touches only that slice
    let mut c = Ndarr::new(&make_i32(&[3, 3], 0), [3usize, 3]).unwrap();
    {
        let mut mid = c.index_axis_mut(1, 1).unwrap();
        mid.map_in_place(|x| x + 100);
    }
    assert_eq!(c.data(), &[0, 101, 2, 3, 104, 5, 6, 107, 8]);
}

#[test]
fn partial_eq_compares_logical_elements_across_backends() {
    let shape = [2usize, 3];
    let data = make_i32(&shape, 0);
    let a = a_u2(&data, shape);

    assert!(a == a.view(), "owned == its own view");
    assert!(a == a.clone());

    // shape mismatch is inequality, not a panic
    let other = a_u2(&make_i32(&[3, 2], 0), [3, 2]);
    assert!(a != other);
    assert_eq!(a.t_view().shape(), other.shape());
    assert!(
        a.t_view() != other.view(),
        "same shape, different element order"
    );

    let mut bumped = a.clone();
    bumped.data_mut()[4] += 1;
    assert!(a != bumped);

    assert!(a.t_view().t_view() == a.view());

    let untyped = a.index_axis_view(0, 1).unwrap();
    let typed = a.index_axis_view(0, 1).unwrap();
    assert!(untyped == typed);
}

#[test]
fn zero_axis_shapes_do_not_panic() {
    for shape in [[3usize, 0], [0, 3], [0, 0]] {
        let a: Ndarr<i32, U2> = Ndarr::new(&[], shape).unwrap();
        assert_eq!(a.len(), 0, "shape={shape:?}");
        assert!(a.is_empty(), "shape={shape:?}");
        assert_eq!(a.iter_elems().count(), 0, "shape={shape:?}");
        assert_eq!(a.to_owned_array().data(), &[] as &[i32]);
        assert_eq!(a.map(|x| x + 1).data(), &[] as &[i32]);
        assert_eq!(a.view().iter_elems().count(), 0);

        let mut rev = shape.to_vec();
        rev.reverse();
        assert_eq!(a.t_view().shape(), rev.as_slice(), "shape={shape:?}");
        assert_eq!(a.t().shape(), rev.as_slice(), "shape={shape:?}");
        assert_eq!(a.t().data(), &[] as &[i32], "shape={shape:?}");

        for (axis, &extent) in shape.iter().enumerate() {
            for i in 0..extent {
                let v = a.index_axis_view(axis, i).unwrap();
                assert_eq!(v.len(), 0);
                assert_eq!(v.iter_elems().count(), 0);
                assert_eq!(v.to_owned_array().data(), &[] as &[i32]);
            }
        }
    }
}

#[test]
fn to_owned_array_normalizes_layout() {
    let shape = [2usize, 3, 4];
    let data = make_i32(&shape, 0);
    let a = a_u3(&data, shape);

    let t = a.t_view().to_owned_array();
    assert_eq!(t.offset(), 0);
    assert_eq!(t.strides(), &[6isize, 2, 1]);
    let (t_expected, t_shape) = transpose_expected(&data, &shape);
    assert_eq_slices("t_owned", &t_shape, t.data(), &t_expected);

    let s = a.index_axis_view(2, 3).unwrap().to_owned_array();
    assert_eq!(s.offset(), 0);
    assert_eq!(s.strides(), &[3isize, 1]);
    let (offset, strides, sub_shape) = slice_layout(&shape, 2, 3);
    let s_expected = gather(&data, offset, &strides, &sub_shape);
    assert_eq_slices("slice_owned", &sub_shape, s.data(), &s_expected);

    assert_eq_slices("owned_roundtrip", &shape, a.to_owned_array().data(), &data);
    assert_eq!(a.to_owned_array().strides(), a.strides());
}

#[test]
fn into_data_yields_logical_order() {
    let shape = [3usize, 1, 4];
    let data = make_i32(&shape, -3);
    let a = a_u3(&data, shape);

    assert_eq_slices("into_data", &shape, &a.clone().into_data(), &data);

    assert!(
        Ndarr::<i32, U3>::new(&data, [3usize, 2, 4]).is_err(),
        "element count mismatch must error"
    );

    let (t_expected, t_shape) = transpose_expected(&data, &shape);
    assert_eq_slices(
        "t_flatten",
        &t_shape,
        &a.t_view().to_owned_array().into_data(),
        &t_expected,
    );
}

#[test]
fn reshape_view_matches_owned_reshape() {
    for shape in SHAPES_3 {
        let n = product(&shape);
        let data = make_i32(&shape, 4);
        let a = a_u3(&data, shape);
        for target in [[n, 1usize], [1, n]] {
            let v = a.view().reshape(target).unwrap();
            assert_eq!(v.offset(), 0);
            assert_eq!(v.strides(), c_strides(&target).as_slice());
            assert_view_invariants!(&format!("reshape_view{shape:?}->{target:?}"), &v);
            // reshape of a C-order array is a flat reinterpretation
            assert_eq_slices("reshape_view", &target, v.to_owned_array().data(), &data);
        }
        assert!(a.view().reshape([n + 1, 1]).is_err());
    }
}

#[test]
fn roll_with_negative_and_zero_shifts_matches_reference() {
    for shape in SHAPES_3 {
        let data = make_i32(&shape, 0);
        let a = a_u3(&data, shape);
        for axis in 0..3 {
            for shift in [-7isize, -3, -1, 0, 1, 3, 7] {
                let ra = a.roll(shift, axis);
                let expected = roll_expected(&data, &shape, shift, axis);
                assert_eq_slices(
                    &format!("roll[{shape:?}/{axis}/{shift}]"),
                    &shape,
                    ra.data(),
                    &expected,
                );
                assert_eq!(ra.strides(), c_strides(&shape).as_slice());
                assert_eq!(ra.offset(), 0);
            }
        }
    }
}

#[test]
fn reduce_over_axis_views_matches_reference() {
    for shape in SHAPES_3 {
        let data = make_i32(&shape, 1);
        let a = a_u3(&data, shape);
        for axis in 0..3 {
            let ra = a.reduce(axis, |x, y| x + y).unwrap();
            let expected = reduce_expected(&data, &shape, axis, |x, y| x + y);
            assert_eq_slices(
                &format!("reduce[{shape:?}/{axis}]"),
                ra.shape(),
                ra.data(),
                &expected,
            );
            let na = a.reduce(axis, |x, y| x.max(y)).unwrap();
            let max_expected = reduce_expected(&data, &shape, axis, |x, y| x.max(y));
            assert_eq_slices("reduce_max", na.shape(), na.data(), &max_expected);
            assert_eq!(ra.offset(), 0);
            assert_eq!(ra.strides(), c_strides(ra.shape()).as_slice());
        }
        assert!(a.reduce(3, |x, y| x + y).is_err());
    }
}

#[test]
fn random_view_pipelines_match_reference() {
    let mut g = Lcg::new(SEED);
    for trial in 0..N_RANDOM {
        let shape = [g.usize_in(1, 4), g.usize_in(1, 4), g.usize_in(1, 4)];
        let data: Vec<i32> = (0..product(&shape)).map(|_| g.i32_in(-50, 50)).collect();
        let a = a_u3(&data, shape);

        let axis = g.usize_in(0, 2);
        let index = g.usize_in(0, shape[axis] - 1);

        let sub = a.index_axis_view(axis, index).unwrap();
        let (offset, strides, sub_shape) = slice_layout(&shape, axis, index);
        let sub_expected = gather(&data, offset, &strides, &sub_shape);
        let tag = format!("rand[{trial}] shape={shape:?} axis={axis} i={index}");
        assert_eq!(sub.shape(), sub_shape.as_slice(), "{tag}: shape");
        assert_eq!(sub.offset(), offset, "{tag}: offset");
        assert_eq!(sub.strides(), strides.as_slice(), "{tag}: strides");
        assert_view_invariants!(&tag, &sub);
        assert_eq_slices(
            &tag,
            sub.shape(),
            sub.to_owned_array().data(),
            &sub_expected,
        );

        let tsub = sub.t_view();
        assert_view_invariants!(&format!("{tag}/t"), &tsub);
        let mut rev_strides = strides.clone();
        rev_strides.reverse();
        let mut rev_shape = sub_shape.clone();
        rev_shape.reverse();
        assert_eq_slices(
            &format!("{tag}/t"),
            tsub.shape(),
            tsub.to_owned_array().data(),
            &gather(&data, offset, &rev_strides, &rev_shape),
        );

        let lead = g.usize_in(1, 3);
        let mut target = vec![lead];
        target.extend_from_slice(sub.shape());
        let dim = Dim::<U3>::new(&target).unwrap();
        let bv = sub.broadcast_view_to(&dim).unwrap();
        assert_eq!(bv.strides()[0], 0, "{tag}: leading broadcast stride");
        assert_view_invariants!(&format!("{tag}/bcast"), &bv);
        let bcast_expected: Vec<i32> = (0..lead)
            .flat_map(|_| sub_expected.iter().copied())
            .collect();
        assert_eq_slices(
            &format!("{tag}/bcast"),
            &target,
            bv.to_owned_array().data(),
            &bcast_expected,
        );

        let mut rev = target.clone();
        rev.reverse();
        let mut bcast_strides = vec![0isize];
        bcast_strides.extend_from_slice(&strides);
        bcast_strides.reverse();
        assert_eq_slices(
            &format!("{tag}/bcast_t"),
            &rev,
            bv.t_view().to_owned_array().data(),
            &gather(&data, offset, &bcast_strides, &rev),
        );

        let (t_expected, t_shape) = transpose_expected(&data, &shape);
        assert_eq_slices(&format!("{tag}/t_all"), &t_shape, a.t().data(), &t_expected);
        let ra = a.reduce(axis, |x, y| x - y).unwrap();
        let reduce_exp = reduce_expected(&data, &shape, axis, |x, y| x - y);
        assert_eq_slices(&format!("{tag}/reduce"), ra.shape(), ra.data(), &reduce_exp);
    }
}

#[test]
fn random_float_view_pipelines_match_reference() {
    let mut g = Lcg::new(SEED ^ 0xF10A7);
    for trial in 0..N_RANDOM {
        let shape = [g.usize_in(1, 4), g.usize_in(1, 5)];
        let data: Vec<f64> = (0..product(&shape)).map(|_| g.f64_in(-5.0, 5.0)).collect();
        let a = Ndarr::new(&data, shape).unwrap();

        // views only move elements, so float equality is exact
        let mut rev_strides = c_strides(&shape);
        rev_strides.reverse();
        let mut rev_shape = shape.to_vec();
        rev_shape.reverse();
        assert_eq_slices(
            &format!("randf[{trial}]/t"),
            &rev_shape,
            a.t_view().to_owned_array().data(),
            &gather(&data, 0, &rev_strides, &rev_shape),
        );

        let axis = g.usize_in(0, 1);
        let index = g.usize_in(0, shape[axis] - 1);
        let sub = a.index_axis_view(axis, index).unwrap();
        let (offset, strides, sub_shape) = slice_layout(&shape, axis, index);
        let sub_expected = gather(&data, offset, &strides, &sub_shape);
        assert_eq_slices(
            &format!("randf[{trial}]/slice"),
            sub.shape(),
            sub.to_owned_array().data(),
            &sub_expected,
        );
        let iterated: Vec<f64> = sub.iter_elems().cloned().collect();
        assert_eq_slices(
            &format!("randf[{trial}]/iter"),
            sub.shape(),
            &iterated,
            &sub_expected,
        );

        let target = [3usize, sub.shape()[0]];
        let dim = Dim::<U2>::new(&target).unwrap();
        let bcast_expected: Vec<f64> = (0..3).flat_map(|_| sub_expected.iter().copied()).collect();
        assert_eq_slices(
            &format!("randf[{trial}]/bcast"),
            &target,
            sub.broadcast_view_to(&dim).unwrap().to_owned_array().data(),
            &bcast_expected,
        );
    }
}

#[test]
fn broadcast_to_uses_numpy_semantics() {
    // [2,1] -> [2,3] repeats each row element ([[7,7,7],[9,9,9]]), not tiles the buffer.
    let a = a_u2(&[7, 9], [2, 1]);
    let expected = vec![7, 7, 7, 9, 9, 9];
    assert_eq_slices(
        "broadcast_to",
        &[2, 3],
        a.broadcast_view_to(&Dim::from([2usize, 3]))
            .unwrap()
            .to_owned_array()
            .data(),
        &expected,
    );
}

#[test]
fn u0_is_scalar_and_dyn_is_runtime_rank() {
    assert_eq!(Dim::<U0>::new(&[]).unwrap().as_slice(), &[] as &[usize]);
    assert!(Dim::<U0>::new(&[1]).is_err());

    for shape in [&[][..], &[1][..], &[2, 3, 4][..], &[1, 0, 5, 2][..]] {
        let dim = Dim::<Dyn>::new(shape).unwrap();
        assert_eq!(dim.as_slice(), shape);
        assert_eq!(dim.len(), shape.len());
    }
}

#[test]
fn all_representative_fixed_ranks_use_exact_inline_storage() {
    fn check<R: StaticRank>() {
        let rank = R::FIXED_RANK.unwrap();
        let store = R::Store::<usize>::try_from_slice(&vec![0; rank]).unwrap();
        assert_eq!(store.as_slice().len(), rank);
        assert!(R::Store::<usize>::try_from_slice(&vec![0; rank + 1]).is_none());
        assert_eq!(
            std::mem::size_of::<R::Store<usize>>(),
            rank * std::mem::size_of::<usize>()
        );
    }

    check::<U0>();
    check::<U8>();
    check::<U16>();
    check::<U32>();
    check::<U64>();
}

#[test]
fn fixed_rank_relations_infer_exact_outputs() {
    let cube = Dim::<U3>::new(&[2, 3, 4]).unwrap();
    let plane: Dim<Removed<U3>> = cube.remove_element(1);
    let restored: Dim<U3> = plane.insert_element(1, 3);
    let vector = Dim::<U1>::new(&[4]).unwrap();
    let broadcast: Dim<Broadcasted<U1, U3>> = vector.broadcast_shape(&restored).unwrap();

    assert_eq!(plane.as_slice(), &[2, 4]);
    assert_eq!(restored.as_slice(), &[2, 3, 4]);
    assert_eq!(broadcast.as_slice(), &[2, 3, 4]);
}

#[test]
fn dynamic_relations_keep_dyn() {
    let cube = Dim::<Dyn>::new(&[2, 3, 4]).unwrap();
    let plane: Dim<Dyn> = cube.remove_element(1);
    let restored: Dim<Dyn> = plane.insert_element(1, 3);
    let fixed = Dim::<U1>::new(&[4]).unwrap();
    let left: Dim<Dyn> = restored.broadcast_shape(&fixed).unwrap();
    let right: Dim<Dyn> = fixed.broadcast_shape(&restored).unwrap();

    assert_eq!(left.as_slice(), &[2, 3, 4]);
    assert_eq!(right.as_slice(), &[2, 3, 4]);
}

#[test]
fn dynamic_shape_behavior_matches_fixed_rank() {
    let shapes = [vec![], vec![1], vec![2, 3], vec![2, 1, 4], vec![1, 0, 3]];
    for shape in shapes {
        let dim = Dim::<Dyn>::new(&shape).unwrap();
        assert_eq!(dim.get_number_elements(), product(&shape));
        // Flat access follows the C-order flat->multi-index correspondence.
        let n_elems = product(&shape);
        let data: Vec<i32> = (0..n_elems as i32).collect();
        let arr: Ndarr<i32, Dyn> = Ndarr::new(&data, dim).unwrap();
        for (n, ix) in indices(&shape).iter().enumerate() {
            assert_eq!(arr[ix.as_slice()], data[n], "shape={shape:?} n={n}");
        }
    }
}

#[test]
fn dynamic_broadcast_matches_fixed_rank() {
    let cases: [(&[usize], &[usize], &[usize]); 2] =
        [(&[3, 1], &[1, 4], &[3, 4]), (&[2, 3, 4], &[4], &[2, 3, 4])];
    for (left, right, expected) in cases {
        let out = Dim::<Dyn>::new(left)
            .unwrap()
            .broadcast_shape(&Dim::<Dyn>::new(right).unwrap())
            .unwrap();
        assert_eq!(out.as_slice(), expected);
    }

    assert!(Dim::<Dyn>::new(&[2, 3])
        .unwrap()
        .broadcast_shape(&Dim::<Dyn>::new(&[4, 3]).unwrap())
        .is_err());

    // NumPy semantics: a zero-length axis against a unit axis stays empty.
    let out = Dim::<Dyn>::new(&[1, 0, 3])
        .unwrap()
        .broadcast_shape(&Dim::<Dyn>::new(&[2, 1, 3]).unwrap())
        .unwrap();
    assert_eq!(out.as_slice(), &[2, 0, 3]);
}

#[test]
fn fixed_and_dynamic_axis_views_match_reference() {
    let data: Vec<i32> = (0..24).collect();
    let shape = [2usize, 3, 4];
    let fixed = Ndarr::<i32, U3>::new(&data, shape).unwrap();
    let dynamic = Ndarr::<i32, Dyn>::new(&data, Dim::<Dyn>::new(&shape).unwrap()).unwrap();

    for axis in 0..3 {
        for index in 0..fixed.shape()[axis] {
            let (offset, strides, sub_shape) = slice_layout(&shape, axis, index);
            let expected = gather(&data, offset, &strides, &sub_shape);
            let fixed_view = fixed.index_axis_view(axis, index).unwrap();
            let dynamic_view = dynamic.index_axis_view(axis, index).unwrap();
            assert_eq!(fixed_view.shape(), sub_shape.as_slice());
            assert_eq!(dynamic_view.shape(), sub_shape.as_slice());
            assert_eq!(
                fixed_view.iter_elems().cloned().collect::<Vec<_>>(),
                expected
            );
            assert_eq!(
                dynamic_view.iter_elems().cloned().collect::<Vec<_>>(),
                expected
            );
        }
    }
}

#[test]
fn const_array_construction_infers_rank_beyond_old_table() {
    let dim: Dim<U3> = [2, 3, 4].into();
    assert_eq!(dim.as_slice(), &[2, 3, 4]);

    let shape = [1usize; 32];
    let high: Dim<U32> = shape.into();
    assert_eq!(high.len(), 32);
}

#[test]
fn rank_metadata_size_is_explicit() {
    assert_eq!(
        std::mem::size_of::<Dim<U8>>(),
        8 * std::mem::size_of::<usize>()
    );
    assert_eq!(
        std::mem::size_of::<Dim<U16>>(),
        16 * std::mem::size_of::<usize>()
    );
    assert_eq!(
        std::mem::size_of::<Dim<U32>>(),
        32 * std::mem::size_of::<usize>()
    );
    assert_eq!(
        std::mem::size_of::<Dim<U64>>(),
        64 * std::mem::size_of::<usize>()
    );
    assert_eq!(
        std::mem::size_of::<Dim<Dyn>>(),
        std::mem::size_of::<Vec<usize>>()
    );
}
