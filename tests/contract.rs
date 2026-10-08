//! Rank-general `contract::<K>` tests vs a brute-force oracle: one left fold of `g ∘ f` per output element, row-major.

use rapl::{s, ContractAxes, Dim, Dyn, Ndarr, StaticRank};
use typenum::{Unsigned, U0, U1, U2, U3, U4};

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

/// Brute-force reference: one left fold over the contracted block, row-major; 1-extent contracted axes broadcast.
fn ref_contract_flat(
    k: usize,
    sa: &[usize],
    da: &[i32],
    sb: &[usize],
    db: &[i32],
    f: impl Fn(&i32, &i32) -> i32,
    g: impl Fn(i32, i32) -> i32,
) -> (Vec<usize>, Vec<i32>) {
    let m = sa.len();
    let ce: Vec<usize> = (0..k).map(|t| sa[m - k + t].max(sb[t])).collect();
    let mut os: Vec<usize> = sa[..m - k].to_vec();
    os.extend_from_slice(&sb[k..]);
    let mut out = Vec::with_capacity(os.iter().product());
    for_each_index(&os, |oidx| {
        let (i_, j_) = oidx.split_at(m - k);
        let mut vals: Vec<i32> = Vec::new();
        for_each_index(&ce, |q| {
            let mut ia = i_.to_vec();
            ia.extend((0..k).map(|t| if sa[m - k + t] == 1 { 0 } else { q[t] }));
            let mut ib: Vec<usize> = (0..k).map(|t| if sb[t] == 1 { 0 } else { q[t] }).collect();
            ib.extend_from_slice(j_);
            vals.push(f(&da[flat(sa, &ia)], &db[flat(sb, &ib)]));
        });
        out.push(vals.into_iter().reduce(&g).unwrap());
    });
    (os, out)
}

/// Deterministic pseudo-random data in -5..=5.
fn seq(shape: &[usize], salt: i32) -> Vec<i32> {
    let n: usize = shape.iter().product();
    (0..n).map(|i| ((i as i32 * 7 + salt) % 11) - 5).collect()
}

// Wrapping arithmetic: fold order is fixed, so wrapped values compare exactly even for the non-associative reducer.
fn nc_f(x: &i32, y: &i32) -> i32 {
    x.wrapping_sub(y.wrapping_mul(2))
}
fn nc_g(x: i32, y: i32) -> i32 {
    x.wrapping_mul(2).wrapping_sub(y)
}
fn mul(x: &i32, y: &i32) -> i32 {
    x.wrapping_mul(*y)
}
fn add(x: i32, y: i32) -> i32 {
    x.wrapping_add(y)
}

/// Check one case against the oracle: all four owned/transposed operand combos, both reducers.
fn check_typed<M, N, K>(sa: &[usize], sb: &[usize])
where
    K: Unsigned + Default + ContractAxes<M, N>,
    M: StaticRank,
    N: StaticRank,
{
    let k = K::USIZE;
    let da = seq(sa, 4);
    let db = seq(sb, 11);

    // Parents hold the transpose contiguously, so t_view() is the canonical array via non-contiguous strides.
    let transposed_parent = |shape: &[usize], data: &[i32]| {
        let tshape: Vec<usize> = shape.iter().rev().cloned().collect();
        let mut tdata = vec![0i32; data.len()];
        for_each_index(shape, |idx| {
            let ridx: Vec<usize> = idx.iter().rev().cloned().collect();
            tdata[flat(&tshape, &ridx)] = data[flat(shape, idx)];
        });
        (tshape, tdata)
    };

    let a = Ndarr::<i32, M>::from_vec_dim(da.clone(), Dim::<M>::new(sa).unwrap()).unwrap();
    let b = Ndarr::<i32, N>::from_vec_dim(db.clone(), Dim::<N>::new(sb).unwrap()).unwrap();
    let (tsa, tda) = transposed_parent(sa, &da);
    let (tsb, tdb) = transposed_parent(sb, &db);
    let ta = Ndarr::<i32, M>::from_vec_dim(tda, Dim::<M>::new(&tsa).unwrap()).unwrap();
    let tb = Ndarr::<i32, N>::from_vec_dim(tdb, Dim::<N>::new(&tsb).unwrap()).unwrap();

    for (f, g) in [
        (mul as fn(&i32, &i32) -> i32, add as fn(i32, i32) -> i32),
        (nc_f, nc_g),
    ] {
        let (os, od) = ref_contract_flat(k, sa, &da, sb, &db, f, g);
        for a_transposed in [false, true] {
            for b_transposed in [false, true] {
                let tag = format!("k={k} sa={sa:?}(t={a_transposed}) sb={sb:?}(t={b_transposed})");
                let result = match (a_transposed, b_transposed) {
                    (false, false) => a.contract(&b, K::default(), f, g).unwrap(),
                    (true, false) => ta.t_view().contract(&b, K::default(), f, g).unwrap(),
                    (false, true) => a.contract(&tb.t_view(), K::default(), f, g).unwrap(),
                    (true, true) => ta
                        .t_view()
                        .contract(&tb.t_view(), K::default(), f, g)
                        .unwrap(),
                };
                assert_eq!(result.shape(), os.as_slice(), "shape {tag}");
                assert_eq!(result.data(), od, "data {tag}");
            }
        }
    }
}

/// Runtime rank and count versions use the same oracle and strided layouts.
fn check_dynamic(k: usize, sa: &[usize], sb: &[usize]) {
    let da = seq(sa, 4);
    let db = seq(sb, 11);
    let a = Ndarr::<_, Dyn>::new(&da, Dim::new(sa).unwrap()).unwrap();
    let b = Ndarr::<_, Dyn>::new(&db, Dim::new(sb).unwrap()).unwrap();
    let at = a.t();
    let bt = b.t();
    for (f, g) in [
        (mul as fn(&i32, &i32) -> i32, add as fn(i32, i32) -> i32),
        (nc_f, nc_g),
    ] {
        let (shape, data) = ref_contract_flat(k, sa, &da, sb, &db, f, g);
        for left in [a.view(), at.t_view()] {
            for right in [b.view(), bt.t_view()] {
                let runtime: Ndarr<i32, Dyn> = left.contract(&right, k, f, g).unwrap();
                let typed: Ndarr<i32, Dyn> = match k {
                    0 => left.contract(&right, U0::new(), f, g),
                    1 => left.contract(&right, U1::new(), f, g),
                    2 => left.contract(&right, U2::new(), f, g),
                    3 => left.contract(&right, U3::new(), f, g),
                    4 => left.contract(&right, U4::new(), f, g),
                    _ => unreachable!(),
                }
                .unwrap();
                assert_eq!(runtime.shape(), shape);
                assert_eq!(runtime.data(), data);
                assert_eq!(typed, runtime);
            }
        }
    }
}

/// Dispatch runtime `(m, n, k)` to `check_typed`: one arm per rank combo in 0..=4.
fn check_case(k: usize, sa: &[usize], sb: &[usize]) {
    check_dynamic(k, sa, sb);
    let (m, n) = (sa.len(), sb.len());
    macro_rules! arms {
        ($(($mv:literal, $nv:literal, $kv:literal, $M:ty, $N:ty, $K:ty)),* $(,)?) => {
            $(
                if m == $mv && n == $nv && k == $kv {
                    return check_typed::<$M, $N, $K>(sa, sb);
                }
            )*
        };
    }
    arms![
        (0, 0, 0, U0, U0, U0),
        (0, 1, 0, U0, U1, U0),
        (0, 2, 0, U0, U2, U0),
        (0, 3, 0, U0, U3, U0),
        (0, 4, 0, U0, U4, U0),
        (1, 0, 0, U1, U0, U0),
        (1, 1, 0, U1, U1, U0),
        (1, 1, 1, U1, U1, U1),
        (1, 2, 0, U1, U2, U0),
        (1, 2, 1, U1, U2, U1),
        (1, 3, 0, U1, U3, U0),
        (1, 3, 1, U1, U3, U1),
        (1, 4, 0, U1, U4, U0),
        (1, 4, 1, U1, U4, U1),
        (2, 0, 0, U2, U0, U0),
        (2, 1, 0, U2, U1, U0),
        (2, 1, 1, U2, U1, U1),
        (2, 2, 0, U2, U2, U0),
        (2, 2, 1, U2, U2, U1),
        (2, 2, 2, U2, U2, U2),
        (2, 3, 0, U2, U3, U0),
        (2, 3, 1, U2, U3, U1),
        (2, 3, 2, U2, U3, U2),
        (2, 4, 0, U2, U4, U0),
        (2, 4, 1, U2, U4, U1),
        (2, 4, 2, U2, U4, U2),
        (3, 0, 0, U3, U0, U0),
        (3, 1, 0, U3, U1, U0),
        (3, 1, 1, U3, U1, U1),
        (3, 2, 0, U3, U2, U0),
        (3, 2, 1, U3, U2, U1),
        (3, 2, 2, U3, U2, U2),
        (3, 3, 0, U3, U3, U0),
        (3, 3, 1, U3, U3, U1),
        (3, 3, 2, U3, U3, U2),
        (3, 3, 3, U3, U3, U3),
        (3, 4, 0, U3, U4, U0),
        (3, 4, 1, U3, U4, U1),
        (3, 4, 2, U3, U4, U2),
        (3, 4, 3, U3, U4, U3),
        (4, 0, 0, U4, U0, U0),
        (4, 1, 0, U4, U1, U0),
        (4, 1, 1, U4, U1, U1),
        (4, 2, 0, U4, U2, U0),
        (4, 2, 1, U4, U2, U1),
        (4, 2, 2, U4, U2, U2),
        (4, 3, 0, U4, U3, U0),
        (4, 3, 1, U4, U3, U1),
        (4, 3, 2, U4, U3, U2),
        (4, 3, 3, U4, U3, U3),
        (4, 4, 0, U4, U4, U0),
        (4, 4, 1, U4, U4, U1),
        (4, 4, 2, U4, U4, U2),
        (4, 4, 3, U4, U4, U3),
        (4, 4, 4, U4, U4, U4),
    ];
    unreachable!("no dispatch arm for m={m}, n={n}, k={k}")
}

/// Every `(m, n, k)` in ranks 0..=4, plus 1-extent variants on contracted and free axes.
fn case_matrix() -> Vec<(usize, Vec<usize>, Vec<usize>)> {
    let mut cases = Vec::new();
    let pool = [2usize, 3, 2, 3];
    for m in 0..=4usize {
        for n in 0..=4usize {
            for k in 0..=m.min(n) {
                let sa: Vec<usize> = (0..m).map(|i| pool[i % pool.len()]).collect();
                let mut sb: Vec<usize> = (0..n).map(|i| pool[(i + 1) % pool.len()]).collect();
                for t in 0..k {
                    sb[t] = sa[m - k + t];
                }
                cases.push((k, sa.clone(), sb.clone()));

                if k > 0 {
                    let mut sa1 = sa.clone();
                    sa1[m - k] = 1;
                    cases.push((k, sa1, sb.clone()));
                    let mut sb1 = sb.clone();
                    sb1[k - 1] = 1;
                    let mut sa2 = sa.clone();
                    if m > k {
                        sa2[0] = 1;
                    }
                    cases.push((k, sa2, sb1));
                } else if m + n > 0 {
                    let mut sa1 = sa.clone();
                    let mut sb1 = sb.clone();
                    if m > 0 {
                        sa1[m - 1] = 1;
                    }
                    if n > 0 {
                        sb1[0] = 1;
                    }
                    cases.push((0, sa1, sb1));
                }
            }
        }
    }
    cases
}

#[test]
fn contract_matches_flat_oracle_across_all_ranks_and_transposes() {
    for (k, sa, sb) in case_matrix() {
        check_case(k, &sa, &sb);
    }
}

#[test]
fn contract_through_strided_and_offset_views() {
    // Negative-step slice: contracting the reversed view equals contracting its materialization.
    let a = Ndarr::from_vec_dim(seq(&[3, 4], 2), Dim::<U2>::new(&[3, 4]).unwrap()).unwrap();
    let b = Ndarr::from_vec_dim(seq(&[4, 5], 9), Dim::<U2>::new(&[4, 5]).unwrap()).unwrap();
    let rev = a.slice(s![.., ..;-1]).unwrap();
    assert_eq!(
        rev.contract(&b, U1::default(), nc_f, nc_g).unwrap(),
        rev.to_owned_array()
            .contract(&b, U1::default(), nc_f, nc_g)
            .unwrap()
    );

    // Offset slice: an interior window of a larger parent.
    let parent = Ndarr::from_vec_dim(seq(&[4, 5], 7), Dim::<U2>::new(&[4, 5]).unwrap()).unwrap();
    let inner = parent.slice(s![1..4, 1..5]).unwrap();
    assert_eq!(inner.shape(), &[3, 4]);
    assert_eq!(
        inner.contract(&b, U1::default(), nc_f, nc_g).unwrap(),
        inner
            .to_owned_array()
            .contract(&b, U1::default(), nc_f, nc_g)
            .unwrap()
    );

    // Strided position slice (step 2), rank 3, k = 2.
    let big = Ndarr::from_vec_dim(seq(&[4, 3, 4], 1), Dim::<U3>::new(&[4, 3, 4]).unwrap()).unwrap();
    let strided = big.slice(s![..;2, .., ..]).unwrap();
    let rhs = Ndarr::from_vec_dim(seq(&[3, 4, 2], 5), Dim::<U3>::new(&[3, 4, 2]).unwrap()).unwrap();
    assert_eq!(
        strided.contract(&rhs, U2::default(), nc_f, nc_g).unwrap(),
        strided
            .to_owned_array()
            .contract(&rhs, U2::default(), nc_f, nc_g)
            .unwrap()
    );
}

#[test]
fn contract_applies_f_and_g_exactly() {
    // Peak-memory proxy: fused traversal calls `f` once per output×block element, `g` block−1 times per output.
    use std::cell::Cell;
    let sa = [2usize, 1, 4];
    let sb = [3usize, 4, 5];
    let a = Ndarr::from_vec_dim(seq(&sa, 1), Dim::<U3>::new(&sa).unwrap()).unwrap();
    let b = Ndarr::from_vec_dim(seq(&sb, 8), Dim::<U3>::new(&sb).unwrap()).unwrap();
    let (out_len, block) = (2 * 5, 3 * 4);

    let fc = Cell::new(0usize);
    let gc = Cell::new(0usize);
    let result = a
        .contract(
            &b,
            U2::default(),
            |x: &i32, y: &i32| {
                fc.set(fc.get() + 1);
                x.wrapping_mul(*y)
            },
            |x: i32, y: i32| {
                gc.set(gc.get() + 1);
                x.wrapping_add(y)
            },
        )
        .unwrap();
    assert_eq!(result.shape(), &[2, 5]);
    assert_eq!(fc.get(), out_len * block);
    assert_eq!(gc.get(), out_len * (block - 1));
}

#[test]
fn contract_empty_free_axes_yield_empty_output() {
    let a = Ndarr::from_vec_dim(Vec::new(), Dim::<U2>::new(&[0, 3]).unwrap()).unwrap();
    let b = Ndarr::from_vec_dim(seq(&[3, 2], 0), Dim::<U2>::new(&[3, 2]).unwrap()).unwrap();
    let c = a.contract(&b, U1::default(), mul, add).unwrap();
    assert_eq!(c.shape(), &[0, 2]);
    assert!(c.data().is_empty());
}

#[test]
fn contract_rejects_mismatched_extents() {
    let a = Ndarr::from_vec_dim(seq(&[2, 3], 0), Dim::<U2>::new(&[2, 3]).unwrap()).unwrap();
    let b = Ndarr::from_vec_dim(seq(&[4, 2], 0), Dim::<U2>::new(&[4, 2]).unwrap()).unwrap();
    assert!(a
        .contract(&b, U1::default(), mul, add)
        .unwrap_err()
        .to_string()
        .contains("contracted pair 0 has extents"));
}

#[test]
fn contract_rejects_empty_contracted_axis() {
    let a = Ndarr::from_vec_dim(Vec::new(), Dim::<U2>::new(&[2, 0]).unwrap()).unwrap();
    let b = Ndarr::from_vec_dim(Vec::new(), Dim::<U2>::new(&[0, 3]).unwrap()).unwrap();
    assert!(a
        .contract(&b, U1::default(), mul, add)
        .unwrap_err()
        .to_string()
        .contains("cannot contract an empty axis"));
}

#[test]
fn fixed_and_dynamic_rank_inference_and_wrappers() {
    let a = Ndarr::from([[1, 2, 3], [4, 5, 6]]);
    let b = Ndarr::from([[1, 2], [3, 4], [5, 6]]);
    let ad = a.clone().into_dyn();
    let bd = b.clone().into_dyn();
    let expected = Ndarr::from([[22, 28], [49, 64]]);
    let fixed: Ndarr<i32, U2> = a.contract(&b, U1::new(), mul, add).unwrap();
    assert_eq!(fixed, expected);
    // Every expression must infer the specified result without a rank cast.
    let dynamic: [Ndarr<i32, Dyn>; 7] = [
        a.contract(&b, 1usize, mul, add).unwrap(),
        a.contract(&bd, U1::new(), mul, add).unwrap(),
        ad.contract(&b, U1::new(), mul, add).unwrap(),
        ad.contract(&bd, U1::new(), mul, add).unwrap(),
        a.contract(&bd, 1usize, mul, add).unwrap(),
        ad.contract(&b, 1usize, mul, add).unwrap(),
        ad.contract(&bd, 1usize, mul, add).unwrap(),
    ];
    for result in dynamic {
        assert_eq!(result, expected);
    }
    assert_eq!(a.mat_mul(&bd).unwrap(), expected);
    assert_eq!(ad.mat_mul(&b).unwrap(), expected);
    assert_eq!(ad.inner_product(&bd, mul, add).unwrap(), expected);
    let outer: Ndarr<i32, Dyn> = ad.outer_product(&b, mul).unwrap();
    let fixed_outer: Ndarr<i32, U4> = a.outer_product(&b, mul).unwrap();
    assert_eq!(outer, fixed_outer);
    let scalar: Ndarr<i32, U0> = a.contract(&a, U2::new(), mul, add).unwrap();
    assert_eq!(scalar.scalar(), 91);
    // Static counts impose no artificial small-rank ceiling.
    let high = Ndarr::<i32, typenum::U16>::new(&[7], [1; 16]).unwrap();
    let high_result: Ndarr<i32, typenum::U16> =
        high.contract(&high, typenum::U8::new(), mul, add).unwrap();
    assert_eq!(high_result.data(), [49]);
}

#[test]
fn invalid_contractions_return_errors_before_callbacks() {
    use std::cell::Cell;
    let called = Cell::new(0);
    let f = |_: &i32, _: &i32| {
        called.set(called.get() + 1);
        0
    };
    let g = |_: i32, _: i32| {
        called.set(called.get() + 1);
        0
    };
    let vector = Ndarr::from([1, 2]);
    let dynamic = vector.clone().into_dyn();
    for k in [2usize, usize::MAX] {
        assert!(vector.contract(&vector, k, f, g).is_err());
        assert!(dynamic.contract(&vector, k, f, g).is_err());
        assert!(vector.contract(&dynamic, k, f, g).is_err());
        assert!(dynamic.contract(&dynamic, k, f, g).is_err());
    }
    assert!(dynamic.contract(&dynamic, U2::new(), f, g).is_err());
    let matrix = Ndarr::from([[1, 2], [3, 4]]);
    assert!(matrix.contract(&dynamic, U2::new(), f, g).is_err());
    assert!(dynamic.contract(&matrix, U2::new(), f, g).is_err());
    for (left, right) in [(0, 0), (0, 1), (1, 0), (0, 2), (2, 3)] {
        for free in [0, 2] {
            let a = Ndarr::<i32, U2>::new(&vec![0; free * left], [free, left]).unwrap();
            let b = Ndarr::<i32, U2>::new(&vec![0; right * 2], [right, 2]).unwrap();
            assert!(a.contract(&b, U1::new(), f, g).is_err());
            assert!(a.contract(&b, 1usize, f, g).is_err());
            assert!(a.mat_mul(&b).is_err());
            assert!(a.inner_product(&b, f, g).is_err());
        }
    }
    assert_eq!(called.get(), 0);
}

#[test]
fn scalar_and_empty_runtime_products_do_not_invoke_unused_callbacks() {
    let scalar = Ndarr::new(&[3i32], []).unwrap();
    let result: Ndarr<i32, Dyn> = scalar
        .contract(&scalar, 0usize, mul, |_, _| panic!("no reduction at k=0"))
        .unwrap();
    assert_eq!(result.shape(), []);
    assert_eq!(result.data(), [9]);
    let empty = Ndarr::<i32, U2>::new(&[], [0, 3]).unwrap();
    let rhs = Ndarr::from([[1, 2], [3, 4], [5, 6]]);
    let result = empty
        .contract(&rhs, 1usize, |_, _| -> i32 { panic!("empty output") }, add)
        .unwrap();
    assert_eq!(result.shape(), [0, 2]);
    let result = empty
        .contract(
            &scalar,
            0usize,
            |_, _| -> i32 { panic!("empty output") },
            add,
        )
        .unwrap();
    assert_eq!(result.shape(), [0, 3]);
}

#[test]
fn contraction_overflow_is_recoverable_without_allocating_or_reading() {
    let scalar = Ndarr::new(&[1u64], []).unwrap();
    let f = |_: &u64, _: &u64| -> u64 { panic!("overflow must be validated first") };
    let g = |_: u64, _: u64| -> u64 { panic!("overflow must be validated first") };
    // Broadcast metadata represents enormous logical arrays with just one stored element.
    let huge = scalar
        .broadcast_view_to(&Dim::<U1>::from([usize::MAX]))
        .unwrap();
    let two = scalar.broadcast_view_to(&Dim::<U1>::from([2])).unwrap();
    assert!(huge
        .outer_product(&two, f)
        .unwrap_err()
        .to_string()
        .contains("element count overflow"));
    assert!(huge
        .outer_product(&scalar, f)
        .unwrap_err()
        .to_string()
        .contains("layout overflow"));
    let many = scalar
        .broadcast_view_to(&Dim::<U1>::from([isize::MAX as usize / 8 + 1]))
        .unwrap();
    assert!(many
        .outer_product(&scalar, f)
        .unwrap_err()
        .to_string()
        .contains("allocation size overflow"));
    // Prefix element products and suffix strides must both be representable,
    // even when an eventual zero extent makes the logical output empty.
    let empty = scalar.broadcast_view_to(&Dim::<U1>::from([0])).unwrap();
    let large = scalar
        .broadcast_view_to(&Dim::<U2>::from([isize::MAX as usize, 3]))
        .unwrap();
    assert!(large.outer_product(&empty, f).is_err());
    assert!(empty.outer_product(&large, f).is_err());
    let block = scalar
        .broadcast_view_to(&Dim::<U2>::from([usize::MAX, 2]))
        .unwrap();
    assert!(block
        .contract(&block, U2::new(), f, g)
        .unwrap_err()
        .to_string()
        .contains("element count overflow"));
}

#[test]
fn shared_axes_contract_each_batch_independently() {
    // Non-commutative, non-associative callbacks pin the per-batch fold order.
    let f = |x: &i32, y: &i32| x - 2 * y;
    let g = |x: i32, y: i32| 2 * x - y;
    let a = Ndarr::new(&(0..24).collect::<Vec<_>>(), [2, 3, 4]).unwrap();
    let b = Ndarr::new(&(0..40).map(|v| v % 7 - 3).collect::<Vec<_>>(), [2, 4, 5]).unwrap();
    let batched: Ndarr<i32, U3> = a.contract(&b, (U1::new(), U1::new()), f, g).unwrap();
    assert_eq!(batched.shape(), &[2, 3, 5]);
    for batch in 0..2isize {
        let left = a.slice(s![batch, .., ..]).unwrap();
        let right = b.slice(s![batch, .., ..]).unwrap();
        let expected = left.contract(&right, U1::new(), f, g).unwrap();
        assert_eq!(batched.slice(s![batch, .., ..]).unwrap(), expected);
    }
    let runtime: Ndarr<i32, Dyn> = a.contract(&b, (1usize, 1usize), f, g).unwrap();
    assert_eq!(runtime, batched);
    let dynamic: Ndarr<i32, Dyn> = a
        .clone()
        .into_dyn()
        .contract(&b, (U1::new(), U1::new()), f, g)
        .unwrap();
    assert_eq!(dynamic, batched);
}

#[test]
fn shared_axes_broadcast_and_batch_outer_products() {
    let mul = |x: &i32, y: &i32| x * y;
    let add = |x: i32, y: i32| x + y;
    let a = Ndarr::from([[[1, 2]]]); // [1, 1, 2]: one batch, broadcast
    let b = Ndarr::from([[[1], [10]], [[100], [1000]]]); // [2, 2, 1]
    let out: Ndarr<i32, U3> = a.contract(&b, (U1::new(), U1::new()), mul, add).unwrap();
    assert_eq!(out, Ndarr::from([[[21]], [[2100]]]));

    // No contracted axes: an outer product per batch; the reducer is unused.
    let x = Ndarr::from([[1, 2], [3, 4]]); // [batch, i]
    let y = Ndarr::from([[1, 10], [100, 1000]]); // [batch, j]
    let outer: Ndarr<i32, U3> = x
        .contract(&y, (U1::new(), U0::new()), mul, |_, _| unreachable!())
        .unwrap();
    assert_eq!(
        outer,
        Ndarr::from([[[1, 10], [2, 20]], [[300, 3000], [400, 4000]]])
    );
}

#[test]
fn invalid_shared_axes_fail_before_callbacks() {
    use std::cell::Cell;
    let called = Cell::new(0);
    let f = |_: &i32, _: &i32| {
        called.set(called.get() + 1);
        0
    };
    let g = |_: i32, _: i32| {
        called.set(called.get() + 1);
        0
    };
    let a = Ndarr::<i32, U3>::new(&[0; 24], [2, 3, 4]).unwrap();
    let b = Ndarr::<i32, U3>::new(&[0; 60], [3, 4, 5]).unwrap();
    assert!(a.contract(&b, (1usize, 1usize), f, g).is_err()); // batches 2 and 3
    assert!(a.contract(&b, (U1::new(), U1::new()), f, g).is_err());
    assert!(a.contract(&a, (3usize, 1usize), f, g).is_err()); // beyond the rank
    assert!(a.contract(&a, (usize::MAX, 1usize), f, g).is_err()); // count overflow
    let dynamic = a.clone().into_dyn();
    assert!(dynamic
        .contract(&dynamic, (U3::new(), U1::new()), f, g)
        .is_err());
    // An empty batch still validates, then produces empty output.
    let empty_a = Ndarr::<i32, U3>::new(&[], [0, 3, 4]).unwrap();
    let empty_b = Ndarr::<i32, U3>::new(&[], [0, 4, 5]).unwrap();
    let empty = empty_a
        .contract(&empty_b, (U1::new(), U1::new()), f, g)
        .unwrap();
    assert_eq!(empty.shape(), &[0, 3, 5]);
    assert_eq!(called.get(), 0);
}
