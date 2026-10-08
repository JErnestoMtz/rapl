use super::*;
use crate::{BroadcastRank, Broadcasted, ContractAxes, Contracted, Dim, Dyn, FrameRank};
use std::ops::*;
use typenum::{U0, U1, U2};

impl<T1, R1: Rank, B1: Buffer<T1>> Ndarr<T1, R1, B1> {
    /// Combine matching elements of two arrays with `f` into a new array.
    /// Shapes broadcast to their common shape, unlike `Iterator::zip`, which
    /// stops at the shorter input; incompatible shapes return an error. The
    /// operands may hold different element types.
    ///
    /// ```
    /// use rapl::*;
    /// let grid = Ndarr::from([[1, 2, 3], [4, 5, 6]]);
    /// let scale = Ndarr::from([10, 100, 1000]); // broadcast down the rows
    /// let scaled = grid.zip_with(&scale, |x, s| x * s)?;
    /// assert_eq!(scaled, Ndarr::from([[10, 200, 3000], [40, 500, 6000]]));
    /// assert!(grid.zip_with(&Ndarr::from([1, 2]), |x, y| x + y).is_err());
    /// # Ok::<(), DimError>(())
    /// ```
    pub fn zip_with<F, T2, B2, T3, R2: Rank>(
        &self,
        other: &Ndarr<T2, R2, B2>,
        f: F,
    ) -> Result<Ndarr<T3, Broadcasted<R1, R2>>, DimError>
    where
        R1: BroadcastRank<R2>,
        B2: Buffer<T2>,
        F: Fn(&T1, &T2) -> T3,
    {
        let new_shape = self.dim.broadcast_shape(&other.dim)?;
        let v1 = self.broadcast_view_to(&new_shape)?;
        let v2 = other.broadcast_view_to(&new_shape)?;
        let new_data: Vec<T3> = v1
            .iter_elems()
            .zip(v2.iter_elems())
            .map(|(a, b)| f(a, b))
            .collect();
        Ok(Ndarr::contiguous(new_data, new_shape))
    }
}

fn checked_len(shape: &[usize]) -> Result<usize, DimError> {
    shape
        .iter()
        .try_fold(1usize, |n, &extent| n.checked_mul(extent))
        .ok_or_else(|| DimError::new("contract: element count overflow"))
}

/// Advance a row-major odometer one step, updating two buffer positions with independent strides.
#[inline]
fn dual_step(
    idx: &mut [usize],
    shape: &[usize],
    pa: &mut isize,
    sta: &[isize],
    pb: &mut isize,
    stb: &[isize],
) {
    for i in (0..shape.len()).rev() {
        idx[i] += 1;
        *pa += sta[i];
        *pb += stb[i];
        if idx[i] < shape[i] {
            return;
        }
        idx[i] = 0;
        *pa -= shape[i] as isize * sta[i];
        *pb -= shape[i] as isize * stb[i];
    }
}

impl<T1, R1: Rank, B1: Buffer<T1>> Ndarr<T1, R1, B1> {
    /// Tensor contraction over `K` axis pairs (NumPy's `tensordot(a, b, axes=k)`),
    /// fused into one pass with no rank `m + n - k` intermediate. Paired elements
    /// are combined with `f` and each contracted block is left-folded with `g` in
    /// row-major order; this fold order is part of the contract. `K` is passed as
    /// a typenum value or runtime `usize`: `U0` is
    /// [`outer_product`](Self::outer_product), `U1` is [`inner_product`](Self::inner_product).
    /// Contracted extents of 1 broadcast against the paired axis.
    ///
    /// A count `(s, k)` first shares the `s` leading axes of both operands, a
    /// batch the two walk together (1-extents broadcast), and contracts `k`
    /// pairs after them: `[S.., free_a.., K..] · [S.., K.., free_b..] ->
    /// [S.., free_a.., free_b..]`. [`mat_mul`](Self::mat_mul) is the
    /// `(rank - 2, 1)` case:
    /// ```
    /// use rapl::*;
    /// let q = Ndarr::from([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]); // [batch, i, k]
    /// let k = Ndarr::from([[[1, 0], [0, 1]], [[2, 0], [0, 2]]]); // [batch, k, j]
    /// let scores: Ndarr<i32, U3> =
    ///     q.contract(&k, (U1::new(), U1::new()), |x, y| x * y, |a, b| a + b)?;
    /// assert_eq!(scores, Ndarr::from([[[1, 2], [3, 4]], [[10, 12], [14, 16]]]));
    /// # Ok::<(), DimError>(())
    /// ```
    ///
    /// Errors on invalid counts, incompatible or empty contracted axes, and
    /// unrepresentable output sizes. A `usize` count or dynamic operand gives
    /// `Dyn` output:
    /// ```
    /// use rapl::*;
    /// let a = Ndarr::from([[1, 2], [3, 4]]);
    /// let fixed: Ndarr<i32, U2> = a.contract(&a, U1::new(), |x, y| x * y, |a, b| a + b)?;
    /// let runtime: Ndarr<i32, Dyn> = a.contract(&a, 1usize, |x, y| x * y, |a, b| a + b)?;
    /// assert_eq!(fixed, runtime);
    /// # Ok::<(), DimError>(())
    /// ```
    /// Typenum counts larger than a fixed operand's rank fail to compile, even
    /// when the other operand is `Dyn`:
    /// ```compile_fail
    /// use rapl::*;
    /// let a = Ndarr::from([[1]]);
    /// a.contract(&a, (U2::new(), U1::new()), |x, y| x * y, |a, b| a + b);
    /// ```
    /// ```compile_fail
    /// use rapl::*;
    /// let a = Ndarr::from([1]);
    /// let b = Ndarr::from([[[1]]]);
    /// a.contract(&b, U2::new(), |x, y| x * y, |a, b| a + b);
    /// ```
    /// ```compile_fail
    /// use rapl::*;
    /// let a = Ndarr::from([[[1]]]);
    /// let b = Ndarr::from([1]);
    /// a.contract(&b, U2::new(), |x, y| x * y, |a, b| a + b);
    /// ```
    /// ```compile_fail
    /// use rapl::*;
    /// let a = Ndarr::from([1]);
    /// let b = Ndarr::from([[[1]]]).into_dyn();
    /// a.contract(&b, U2::new(), |x, y| x * y, |a, b| a + b);
    /// ```
    /// ```compile_fail
    /// use rapl::*;
    /// let a = Ndarr::from([[[1]]]).into_dyn();
    /// let b = Ndarr::from([1]);
    /// a.contract(&b, U2::new(), |x, y| x * y, |a, b| a + b);
    /// ```
    pub fn contract<K, F, G, T2, B2, T3, R2>(
        &self,
        other: &Ndarr<T2, R2, B2>,
        k: K,
        f: F,
        g: G,
    ) -> Result<Ndarr<T3, Contracted<R1, R2, K>>, DimError>
    where
        K: ContractAxes<R1, R2>,
        R2: Rank,
        B2: Buffer<T2>,
        F: Fn(&T1, &T2) -> T3,
        G: Fn(T3, T3) -> T3,
    {
        let s = k.shared();
        let k = k.count();
        let (sa, sb) = (self.shape(), other.shape());
        let (ta, tb) = (self.strides(), other.strides());
        let (m, n) = (sa.len(), sb.len());
        if s.checked_add(k).is_none_or(|used| used > m || used > n) {
            return Err(DimError::new(&format!(
                "contract: {s} shared and {k} contracted axes exceed operand ranks {m} and {n}"
            )));
        }
        let (fa, fb) = (m - s - k, n - s - k); // free axis counts

        // Shared and contracted axes pair up and broadcast: a 1-extent walks in
        // place with stride 0. Each pair yields its extent and both strides.
        let pair = |kind: &str, t: usize, a: usize, b: usize| {
            let (ea, eb) = (sa[a], sb[b]);
            if ea != eb && ea != 1 && eb != 1 {
                return Err(DimError::new(&format!(
                    "contract: {kind} pair {t} has extents {ea} and {eb}"
                )));
            }
            let walk = |extent, stride| if extent == 1 { 0 } else { stride };
            Ok((
                if ea == 1 { eb } else { ea },
                walk(ea, ta[a]),
                walk(eb, tb[b]),
            ))
        };
        let shared = (0..s)
            .map(|t| pair("shared", t, t, t))
            .collect::<Result<Vec<_>, _>>()?;
        let contracted = (0..k)
            .map(|t| pair("contracted", t, s + fa + t, s + t))
            .collect::<Result<Vec<_>, _>>()?;
        let ce: Vec<usize> = contracted.iter().map(|c| c.0).collect();
        if ce.contains(&0) {
            return Err(DimError::new("contract: cannot contract an empty axis"));
        }
        let block_len = checked_len(&ce)?;
        let ca: Vec<isize> = contracted.iter().map(|c| c.1).collect();
        let cb: Vec<isize> = contracted.iter().map(|c| c.2).collect();

        // Output axes: shared axes move both operands, then each operand's free
        // axes move only that operand.
        let mut os: Vec<usize> = shared.iter().map(|c| c.0).collect();
        let mut oa: Vec<isize> = shared.iter().map(|c| c.1).collect();
        let mut ob: Vec<isize> = shared.iter().map(|c| c.2).collect();
        os.extend_from_slice(&sa[s..s + fa]);
        oa.extend_from_slice(&ta[s..s + fa]);
        ob.extend(std::iter::repeat_n(0, fa));
        os.extend_from_slice(&sb[s + k..]);
        oa.extend(std::iter::repeat_n(0, fb));
        ob.extend_from_slice(&tb[s + k..]);

        let out_len = checked_len(&os)?;
        // Owned outputs use signed strides; reject unrepresentable layouts before allocating.
        os.iter()
            .rev()
            .try_fold(1isize, |stride, &extent| {
                isize::try_from(extent)
                    .ok()
                    .and_then(|extent| stride.checked_mul(extent))
            })
            .ok_or_else(|| DimError::new("contract: output layout overflow"))?;
        out_len
            .checked_mul(std::mem::size_of::<T3>())
            .filter(|&bytes| bytes <= isize::MAX as usize)
            .ok_or_else(|| DimError::new("contract: output allocation size overflow"))?;
        let out_dim = Dim::<Contracted<R1, R2, K>>::new(&os)?;
        let mut out: Vec<T3> = Vec::with_capacity(out_len);
        let (abuf, bbuf) = (self.buffer.as_slice(), other.buffer.as_slice());

        // The only scratch: two odometer index registers.
        let mut oidx = vec![0usize; os.len()];
        let mut cidx = vec![0usize; k];

        let (mut base_a, mut base_b) = (self.offset as isize, other.offset as isize);
        for _ in 0..out_len {
            let (mut pa, mut pb) = (base_a, base_b);
            let mut acc = f(&abuf[pa as usize], &bbuf[pb as usize]);
            for _ in 1..block_len {
                dual_step(&mut cidx, &ce, &mut pa, &ca, &mut pb, &cb);
                acc = g(acc, f(&abuf[pa as usize], &bbuf[pb as usize]));
            }
            cidx.iter_mut().for_each(|v| *v = 0);
            out.push(acc);
            dual_step(&mut oidx, &os, &mut base_a, &oa, &mut base_b, &ob);
        }

        Ok(Ndarr::contiguous(out, out_dim))
    }

    /// Matrix product of the last two axes, broadcasting the leading (batch)
    /// axes like NumPy's `@`: `[.., i, k] × [.., k, j] -> [.., i, j]`. It is
    /// [`contract`](Self::contract) with count `(rank - 2, 1)` after padding the
    /// lower-rank operand with leading unit axes. Both operands need at least
    /// two axes; vectors and `tensordot`-style products use
    /// [`inner_product`](Self::inner_product).
    /// ```
    /// use rapl::*;
    /// let batch = Ndarr::from([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]); // [2, 2, 2]
    /// let double = Ndarr::from([[2, 0], [0, 2]]); // broadcast over the batch
    /// assert_eq!(batch.mat_mul(&double)?, &batch * 2);
    /// # Ok::<(), DimError>(())
    /// ```
    /// ```compile_fail
    /// let vector = rapl::Ndarr::from([1, 2]);
    /// vector.mat_mul(&vector);
    /// ```
    pub fn mat_mul<B2, R2>(
        &self,
        other: &Ndarr<T1, R2, B2>,
    ) -> Result<Ndarr<T1, Broadcasted<R1, R2>>, DimError>
    where
        R1: BroadcastRank<R2> + FrameRank<U2>,
        R2: FrameRank<U2>,
        B2: Buffer<T1>,
        T1: Clone + Add<Output = T1> + Mul<Output = T1>,
    {
        if self.rank() < 2 || other.rank() < 2 {
            return Err(DimError::new(
                "mat_mul: operands need at least two axes; use inner_product for vectors",
            ));
        }
        let rank = self.rank().max(other.rank());
        let lead = |shape: &[usize]| {
            let mut padded = vec![1; rank - shape.len()];
            padded.extend_from_slice(shape);
            Dim::<Dyn>::new(&padded)
        };
        let a = self.broadcast_view_to(&lead(self.shape())?)?;
        let b = other.broadcast_view_to(&lead(other.shape())?)?;
        let mul = |x: &T1, y: &T1| x.clone() * y.clone();
        let product = a.contract(&b, (rank - 2, 1usize), mul, |x, y| x + y)?;
        let dim = Dim::new(product.shape())?;
        Ok(Ndarr::contiguous(product.into_data(), dim))
    }

    pub fn inner_product<F, G, T2, B2, T3, R2>(
        &self,
        other: &Ndarr<T2, R2, B2>,
        f: F,
        g: G,
    ) -> Result<Ndarr<T3, Contracted<R1, R2, U1>>, DimError>
    where
        U1: ContractAxes<R1, R2>,
        R2: Rank,
        B2: Buffer<T2>,
        F: Fn(&T1, &T2) -> T3,
        G: Fn(T3, T3) -> T3,
    {
        self.contract(other, U1::new(), f, g)
    }

    pub fn outer_product<F, T2, B2, T3, R2>(
        &self,
        other: &Ndarr<T2, R2, B2>,
        f: F,
    ) -> Result<Ndarr<T3, Contracted<R1, R2, U0>>, DimError>
    where
        U0: ContractAxes<R1, R2>,
        R2: Rank,
        B2: Buffer<T2>,
        F: Fn(&T1, &T2) -> T3,
    {
        // With no contracted axes the reducer is never called.
        self.contract(other, U0::new(), f, |x, _| x)
    }
}

#[cfg(test)]
mod dyadic_test {
    use super::*;
    use crate::{Dyn, StaticRank};
    use typenum::{Unsigned, U2};

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

    /// Brute-force reference for `contract`, with the same single left fold in
    /// row-major block order, so it matches even for non-associative, non-commutative `g`.
    fn ref_contract<A, B, C>(
        k: usize,
        sa: &[usize],
        da: &[A],
        sb: &[usize],
        db: &[B],
        f: impl Fn(&A, &B) -> C,
        g: impl Fn(C, C) -> C,
    ) -> (Vec<usize>, Vec<C>) {
        let m = sa.len();
        let ce: Vec<usize> = (0..k).map(|t| sa[m - k + t].max(sb[t])).collect();
        let mut os: Vec<usize> = sa[..m - k].to_vec();
        os.extend_from_slice(&sb[k..]);
        let mut out = Vec::with_capacity(os.iter().product());
        for_each_index(&os, |oidx| {
            let (i_, j_) = oidx.split_at(m - k);
            let mut vals: Vec<C> = Vec::new();
            for_each_index(&ce, |q| {
                let mut ia = i_.to_vec();
                ia.extend((0..k).map(|t| if sa[m - k + t] == 1 { 0 } else { q[t] }));
                let mut ib: Vec<usize> =
                    (0..k).map(|t| if sb[t] == 1 { 0 } else { q[t] }).collect();
                ib.extend_from_slice(j_);
                vals.push(f(&da[flat(sa, &ia)], &db[flat(sb, &ib)]));
            });
            out.push(vals.into_iter().reduce(&g).unwrap());
        });
        (os, out)
    }

    /// Equivalence oracle: broadcast views, `zip_with`, then one `reduce` per
    /// contracted axis. Its per-axis fold
    /// only matches the fused fold for associative + commutative `g`.
    fn composed_contract<K, F, G, T1, T2, T3, R1, R2>(
        lhs: &Ndarr<T1, R1>,
        other: &Ndarr<T2, R2>,
        f: F,
        g: G,
    ) -> Ndarr<T3, Contracted<R1, R2, K>>
    where
        K: Unsigned + ContractAxes<R1, R2>,
        R1: StaticRank,
        R2: StaticRank,
        T3: Clone,
        F: Fn(&T1, &T2) -> T3,
        G: Fn(T3, T3) -> T3,
    {
        let k = K::USIZE;
        let m = lhs.dim.len();
        let rank_intermediate = m + other.dim.len() - k;
        let pad_left = |shape: &[usize]| {
            let mut padded = vec![1; rank_intermediate];
            padded[rank_intermediate - shape.len()..].copy_from_slice(shape);
            Dim::<Dyn>::new(&padded).unwrap()
        };
        let t1 = lhs.t_view();
        let b1 = t1.broadcast_view_to(&pad_left(t1.shape())).unwrap();
        let v1 = b1.t_view();
        let v2 = other.broadcast_view_to(&pad_left(other.shape())).unwrap();
        let mut r = v1.zip_with(&v2, f).unwrap();
        for axis in (m - k..m).rev() {
            r = r.reduce(axis, &g).unwrap();
        }
        let out_dim = Dim::<Contracted<R1, R2, K>>::new(r.shape()).unwrap();
        Ndarr::from_vec_dim(r.into_data(), out_dim).unwrap()
    }

    /// Deterministic pseudo-random test data.
    fn seq(shape: &[usize], salt: i32) -> Vec<i32> {
        let n: usize = shape.iter().product();
        (0..n).map(|i| ((i as i32 * 7 + salt) % 11) - 5).collect()
    }

    // Non-commutative, non-associative pair: the fold structure is part of the contract.
    fn nc_f(x: &i32, y: &i32) -> i32 {
        x - 2 * y
    }
    fn nc_g(x: i32, y: i32) -> i32 {
        2 * x - y
    }

    #[test]
    fn contract_k0_equals_outer_product() {
        for (p, q, r) in [(1, 2, 3), (3, 1, 2), (4, 5, 2)] {
            let da = seq(&[p], 1);
            let db = seq(&[q, r], 7);
            let a = Ndarr::new(&da, [p]).unwrap();
            let b = Ndarr::new(&db, [q, r]).unwrap();

            let c = a.contract(&b, U0::new(), nc_f, nc_g).unwrap();
            assert_eq!(c, a.outer_product(&b, nc_f).unwrap());

            let (os, od) = ref_contract(0, &[p], &da, &[q, r], &db, nc_f, nc_g);
            assert_eq!(c.shape(), os.as_slice());
            assert_eq!(c.data(), od);
        }

        let s = Ndarr::new(&[3], []).unwrap();
        let v = Ndarr::from([1, 2, 3]);
        assert_eq!(
            s.contract(&v, U0::new(), nc_f, nc_g).unwrap(),
            s.outer_product(&v, nc_f).unwrap()
        );
    }

    #[test]
    fn contract_k1_equals_inner_product() {
        // (m, k, n) matrix pairs, including broadcasting 1-extents.
        for (m, k1, k2, n) in [(2, 3, 3, 4), (3, 1, 4, 2), (2, 5, 1, 3), (1, 2, 2, 1)] {
            let da = seq(&[m, k1], 3);
            let db = seq(&[k2, n], 5);
            let a = Ndarr::new(&da, [m, k1]).unwrap();
            let b = Ndarr::new(&db, [k2, n]).unwrap();

            let c = a.contract(&b, U1::new(), nc_f, nc_g).unwrap();
            assert_eq!(c, a.inner_product(&b, nc_f, nc_g).unwrap());

            let (os, od) = ref_contract(1, &[m, k1], &da, &[k2, n], &db, nc_f, nc_g);
            assert_eq!(c.shape(), os.as_slice());
            assert_eq!(c.data(), od);
        }

        // Mixed ranks: vector · rank-3 (`tensordot`, not a batched product).
        let da = seq(&[3], 2);
        let db = seq(&[3, 2, 4], 9);
        let a = Ndarr::new(&da, [3]).unwrap();
        let b = Ndarr::new(&db, [3, 2, 4]).unwrap();
        assert_eq!(
            a.contract(&b, U1::new(), nc_f, nc_g).unwrap(),
            a.inner_product(&b, nc_f, nc_g).unwrap()
        );
    }

    #[test]
    fn contract_k2_matches_reference() {
        // Double contraction, including broadcasting 1-extents.
        for (sa, sb) in [
            ([2, 3, 4], [3, 4, 5]),
            ([2, 1, 4], [3, 4, 2]),
            ([3, 2, 4], [2, 1, 3]),
        ] {
            let da = seq(&sa, 4);
            let db = seq(&sb, 11);
            let a = Ndarr::new(&da, sa).unwrap();
            let b = Ndarr::new(&db, sb).unwrap();

            let c = a.contract(&b, U2::new(), nc_f, nc_g).unwrap();
            let (os, od) = ref_contract(2, &sa, &da, &sb, &db, nc_f, nc_g);
            assert_eq!(c.shape(), os.as_slice());
            assert_eq!(c.data(), od);
        }

        // Full contraction with multiply/add is the element-wise product sum.
        let da = seq(&[2, 3], 1);
        let db = seq(&[2, 3], 8);
        let a = Ndarr::new(&da, [2, 3]).unwrap();
        let b = Ndarr::new(&db, [2, 3]).unwrap();
        let dot: i32 = da.iter().zip(&db).map(|(x, y)| x * y).sum();
        assert_eq!(
            a.contract(&b, U2::new(), |x, y| x * y, |x, y| x + y)
                .unwrap()
                .scalar(),
            dot
        );
    }

    #[test]
    fn fused_contract_matches_composition() {
        // Associative + commutative reducer: fused and per-axis folds must agree exactly.
        let mul = |x: &i32, y: &i32| x * y;
        let add = |x: i32, y: i32| x + y;

        let a = Ndarr::new(&seq(&[2, 3, 4], 3), [2, 3, 4]).unwrap();
        let b = Ndarr::new(&seq(&[3, 4, 5], 8), [3, 4, 5]).unwrap();
        assert_eq!(
            a.contract(&b, U2::new(), mul, add).unwrap(),
            composed_contract::<U2, _, _, _, _, _, _, _>(&a, &b, mul, add)
        );

        let c = Ndarr::new(&seq(&[4, 3], 1), [4, 3]).unwrap();
        let d = Ndarr::new(&seq(&[3, 2], 6), [3, 2]).unwrap();
        assert_eq!(
            c.contract(&d, U1::new(), mul, add).unwrap(),
            composed_contract::<U1, _, _, _, _, _, _, _>(&c, &d, mul, add)
        );
        assert_eq!(
            c.contract(&d, U0::new(), mul, add).unwrap(),
            composed_contract::<U0, _, _, _, _, _, _, _>(&c, &d, mul, add)
        );
    }

    #[test]
    fn contract_accepts_views_and_ignores_layout() {
        // A transposed view contracts like the materialized transpose; same for a negative-step slice.
        let a = Ndarr::new(&seq(&[3, 4], 2), [3, 4]).unwrap();
        let b = Ndarr::new(&seq(&[3, 5], 9), [3, 5]).unwrap();
        assert_eq!(
            a.t_view().contract(&b, U1::new(), nc_f, nc_g).unwrap(),
            a.t().contract(&b, U1::new(), nc_f, nc_g).unwrap()
        );

        // [5, 4] reversed then transposed gives a [4, 5] view pairing with a's trailing 4.
        let e = Ndarr::new(&seq(&[5, 4], 6), [5, 4]).unwrap();
        let reversed = e.slice(crate::s![.., ..;-1]).unwrap();
        assert_eq!(
            a.contract(&reversed.t_view(), U1::new(), nc_f, nc_g)
                .unwrap(),
            a.contract(&reversed.to_owned_array().t(), U1::new(), nc_f, nc_g)
                .unwrap()
        );
    }

    #[test]
    fn outer() {
        let z = Ndarr::from([1, 2, 3]);
        let g = |a: &i32, b: &i32| {
            if a == b {
                1
            } else {
                0
            }
        };
        let r1 = z.outer_product(&z, |x, y| x + y).unwrap();
        let r2 = z.outer_product(&z, g).unwrap();
        assert_eq!(r1, Ndarr::from([[2, 3, 4], [3, 4, 5], [4, 5, 6]]));
        assert_eq!(r2, Ndarr::from([[1, 0, 0], [0, 1, 0], [0, 0, 1]]));
    }

    #[test]
    fn mat_mul() {
        let a = Ndarr::from([[0, 1, 2], [3, 4, 5], [6, 7, 8]]);
        let b = Ndarr::from([[9, 10, 11], [12, 13, 14], [15, 16, 17]]);
        let result_a_b = Ndarr::from([[42, 45, 48], [150, 162, 174], [258, 279, 300]]);

        assert_eq!(a.mat_mul(&b).unwrap(), result_a_b);

        let c = Ndarr::from(0..5);
        let dot = c.inner_product(&c, |x, y| x * y, |x, y| x + y).unwrap();
        assert_eq!(dot.scalar(), 30)
    }

    /// Per-matrix reference: every leading index selects two 2-D slices.
    fn mat_mul_by_slices(a: &Ndarr<i32, Dyn>, b: &Ndarr<i32, Dyn>) -> Vec<i32> {
        let (rank, a_rank, b_rank) = (a.rank().max(b.rank()), a.rank(), b.rank());
        let batch: Vec<usize> = (0..rank - 2)
            .map(|axis| {
                let extent = |shape: &[usize], r: usize| {
                    if axis + r >= rank {
                        shape[axis + r - rank]
                    } else {
                        1
                    }
                };
                // Not `max`: a zero extent paired with a unit extent stays zero.
                match (extent(a.shape(), a_rank), extent(b.shape(), b_rank)) {
                    (1, other) | (other, _) => other,
                }
            })
            .collect();
        let mut out = Vec::new();
        for_each_index(&batch, |index| {
            let select = |x: &Ndarr<i32, Dyn>| {
                let lead = rank - x.rank();
                let mut specs: Vec<SliceSpec> = (lead..rank - 2)
                    .map(|axis| {
                        let extent = x.shape()[axis - lead];
                        SliceSpec::Index(if extent == 1 { 0 } else { index[axis] } as isize)
                    })
                    .collect();
                specs.extend([SliceSpec::from(..), SliceSpec::from(..)]);
                x.slice(&specs[..])
                    .unwrap()
                    .to_owned_array()
                    .into_ranked::<U2>()
                    .unwrap()
            };
            out.extend(select(a).mat_mul(&select(b)).unwrap().into_data());
        });
        out
    }

    #[test]
    fn mat_mul_batches_and_broadcasts_leading_axes() {
        for (sa, sb, so) in [
            (&[2, 3, 4][..], &[2, 4, 5][..], &[2, 3, 5][..]),
            (&[3, 2, 4], &[4, 5], &[3, 2, 5]),
            (&[3, 4], &[2, 4, 5], &[2, 3, 5]),
            (&[2, 1, 3, 4], &[3, 4, 2], &[2, 3, 3, 2]),
            (&[1, 3, 3, 4], &[2, 1, 4, 2], &[2, 3, 3, 2]),
            (&[0, 3, 4], &[4, 5], &[0, 3, 5]),
        ] {
            let a = Ndarr::<i32, Dyn>::new(&seq(sa, 1), sa).unwrap();
            let b = Ndarr::<i32, Dyn>::new(&seq(sb, 4), sb).unwrap();
            let product = a.mat_mul(&b).unwrap();
            assert_eq!(product.shape(), so, "{sa:?} @ {sb:?}");
            assert_eq!(product.data(), mat_mul_by_slices(&a, &b), "{sa:?} @ {sb:?}");
        }
        let fixed: Ndarr<i32, typenum::U3> = Ndarr::new(&seq(&[2, 3, 4], 1), [2, 3, 4])
            .unwrap()
            .mat_mul(&Ndarr::new(&seq(&[4, 5], 2), [4, 5]).unwrap())
            .unwrap();
        assert_eq!(fixed.shape(), &[2, 3, 5]);
    }

    #[test]
    fn mat_mul_rejects_mismatched_batches_and_vectors() {
        let a = Ndarr::<i32, Dyn>::new(&seq(&[2, 3, 4], 1), &[2, 3, 4][..]).unwrap();
        let b = Ndarr::<i32, Dyn>::new(&seq(&[3, 4, 5], 1), &[3, 4, 5][..]).unwrap();
        assert!(a.mat_mul(&b).is_err());
        let vector = Ndarr::<i32, Dyn>::new(&[1, 2, 3, 4], &[4][..]).unwrap();
        assert!(a.mat_mul(&vector).is_err());
        assert!(vector.mat_mul(&a).is_err());
    }

    #[test]
    fn inner_product() {
        // `tensordot` semantics: every free axis of both operands survives.
        let a = Ndarr::from(1..13).reshape([2, 2, 3]).unwrap();
        let b = Ndarr::from(1..13).reshape([3, 2, 2]).unwrap();
        let inner = a.inner_product(&b, |x, y| x * y, |x, y| x + y).unwrap();
        assert_eq!(inner.shape(), &[2, 2, 2, 2]);
        assert_eq!(
            inner,
            a.contract(&b, U1::new(), |x, y| x * y, |x, y| x + y)
                .unwrap()
        );
        assert!(a.mat_mul(&b).is_err());
    }
}
