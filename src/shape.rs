use crate::errors::DimError;
use generic_array::{ArrayLength, ConstArrayLength, GenericArray, IntoArrayLength};
use std::fmt::Debug;
use std::marker::PhantomData;
use std::ops::{Add, Sub};
use typenum::{Add1, Const, Max, Maximum, Unsigned, B1, U0, U1};

/// A rank whose number of axes is known only at runtime.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Dyn;

/// Storage used for the shape or strides selected by a [`Rank`].
pub trait RankStore<T>: Clone + Debug {
    /// `None` when the slice length does not fit the store's fixed rank.
    fn try_from_slice(slice: &[T]) -> Option<Self>;
    fn as_slice(&self) -> &[T];
}

impl<T, R> RankStore<T> for GenericArray<T, R>
where
    T: Copy + Default + Debug,
    R: ArrayLength,
{
    fn try_from_slice(slice: &[T]) -> Option<Self> {
        if slice.len() != R::USIZE {
            return None;
        }
        let mut result = Self::default();
        GenericArray::as_mut_slice(&mut result).copy_from_slice(slice);
        Some(result)
    }

    fn as_slice(&self) -> &[T] {
        GenericArray::as_slice(self)
    }
}

impl<T> RankStore<T> for Vec<T>
where
    T: Copy + Default + Debug,
{
    fn try_from_slice(slice: &[T]) -> Option<Self> {
        Some(slice.to_vec())
    }

    fn as_slice(&self) -> &[T] {
        self
    }
}

/// Selects storage for per-axis metadata. Every typenum rank uses an exact
/// inline [`GenericArray`]; only [`Dyn`] uses a `Vec`, so fixed rank and
/// metadata length cannot disagree.
#[diagnostic::on_unimplemented(
    message = "`{Self}` is not a rank",
    label = "expected a typenum unsigned rank (`U0`, `U1`, ...) or `Dyn`"
)]
pub trait Rank {
    /// The structural rank, or `None` for [`Dyn`].
    const FIXED_RANK: Option<usize>;
    type Store<T: Copy + Default + Debug>: RankStore<T>;
}

impl<R> Rank for R
where
    R: ArrayLength,
{
    const FIXED_RANK: Option<usize> = Some(R::USIZE);
    type Store<T: Copy + Default + Debug> = GenericArray<T, R>;
}

impl Rank for Dyn {
    const FIXED_RANK: Option<usize> = None;
    type Store<T: Copy + Default + Debug> = Vec<T>;
}

/// A fixed typenum rank. Algorithms using typenum arithmetic may require this
/// stronger bound; ordinary structural APIs should prefer [`Rank`].
#[diagnostic::on_unimplemented(
    message = "`{Self}` is not a fixed rank",
    label = "this operation does rank arithmetic and needs a typenum rank (`U0`, `U1`, ...), not `Dyn`"
)]
pub trait StaticRank: Rank + Unsigned {}
impl<R> StaticRank for R where R: Rank + Unsigned {}

#[diagnostic::on_unimplemented(
    message = "cannot remove an axis from rank `{Self}`",
    label = "expected a typenum rank of at least `U1`, or `Dyn`"
)]
pub trait RemoveAxisRank: Rank {
    type Output: Rank;
}

/// Removing any one axis has the rank arithmetic of a one-axis frame.
impl<R: FrameRank<U1>> RemoveAxisRank for R {
    type Output = <R as FrameRank<U1>>::Output;
}

#[diagnostic::on_unimplemented(
    message = "cannot insert an axis into rank `{Self}`",
    label = "expected a typenum rank (`U0`, `U1`, ...) or `Dyn`"
)]
pub trait InsertAxisRank: Rank {
    type Output: Rank;
}

impl<R> InsertAxisRank for R
where
    R: StaticRank + Add<B1, Output: StaticRank>,
{
    type Output = Add1<R>;
}

impl InsertAxisRank for Dyn {
    type Output = Dyn;
}

#[diagnostic::on_unimplemented(
    message = "cannot broadcast rank `{Self}` with rank `{Rhs}`",
    label = "both ranks must be typenum ranks (`U0`, `U1`, ...) or `Dyn`"
)]
pub trait BroadcastRank<Rhs: Rank>: Rank {
    type Output: Rank;
}

impl<L, R> BroadcastRank<R> for L
where
    L: StaticRank + Max<R, Output: StaticRank>,
    R: StaticRank,
{
    type Output = Maximum<L, R>;
}

impl<R: StaticRank> BroadcastRank<R> for Dyn {
    type Output = Dyn;
}

impl<L: StaticRank> BroadcastRank<Dyn> for L {
    type Output = Dyn;
}

impl BroadcastRank<Dyn> for Dyn {
    type Output = Dyn;
}

pub type Removed<R> = <R as RemoveAxisRank>::Output;
pub type Inserted<R> = <R as InsertAxisRank>::Output;
pub type Broadcasted<L, R> = <L as BroadcastRank<R>>::Output;

/// Rank resulting from contracting `K` axis pairs while sharing `S` leading
/// axes (`L + R - S - 2K`). Fixed operands retain compile-time count guards;
/// a dynamic operand makes the output dynamic.
pub trait ContractRank<Rhs: Rank, K: Unsigned, S: Unsigned = U0>: Rank {
    type Output: Rank;
}

impl<L, R, K, S> ContractRank<R, K, S> for L
where
    L: StaticRank + Add<R, Output: Sub<K, Output: Sub<K, Output: Sub<S, Output: StaticRank>>>>,
    R: StaticRank,
    K: Unsigned
        + Add<
            S,
            Output: typenum::IsLessOrEqual<L, Output = typenum::True>
                        + typenum::IsLessOrEqual<R, Output = typenum::True>,
        >,
    S: Unsigned,
{
    type Output = typenum::Diff<typenum::Diff<typenum::Diff<typenum::Sum<L, R>, K>, K>, S>;
}

impl<R, K, S> ContractRank<R, K, S> for Dyn
where
    R: StaticRank,
    K: Unsigned + Add<S, Output: typenum::IsLessOrEqual<R, Output = typenum::True>>,
    S: Unsigned,
{
    type Output = Dyn;
}

impl<L, K, S> ContractRank<Dyn, K, S> for L
where
    L: StaticRank,
    K: Unsigned + Add<S, Output: typenum::IsLessOrEqual<L, Output = typenum::True>>,
    S: Unsigned,
{
    type Output = Dyn;
}

impl<K: Unsigned, S: Unsigned> ContractRank<Dyn, K, S> for Dyn {
    type Output = Dyn;
}

mod axis_count_sealed {
    pub trait Sealed {}
    impl Sealed for usize {}
    impl Sealed for typenum::UTerm {}
    impl<U, B> Sealed for typenum::UInt<U, B> {}
    impl Sealed for (usize, usize) {}
    impl<S: Typenum, K: Typenum> Sealed for (S, K) {}
    impl Sealed for super::Keep {}

    /// Typenum counts, as distinct from `usize` ones.
    pub trait Typenum: typenum::Unsigned {}
    impl Typenum for typenum::UTerm {}
    impl<U: typenum::Unsigned, B: typenum::Bit> Typenum for typenum::UInt<U, B> {}
}

/// A reduction axis that stays in the result with extent 1, so the result
/// broadcasts back against its input: NumPy's `keepdims=True`. A bare `usize`
/// axis is removed instead. The axis is named once, so the reduction and the
/// re-inserted axis cannot disagree.
///
/// ```
/// use rapl::*;
/// use std::ops::Add;
/// let scores = Ndarr::from([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]]);
/// let exp = (&scores - &scores.reduce(Keep(1), f64::max)?).exp(); // max: [2, 1]
/// let softmax = &exp / &exp.reduce(Keep(1), f64::add)?;
/// assert!(softmax.reduce(1, f64::add)?.approx_epsilon(&Ndarr::from([1.0, 1.0]), 1e-12));
/// # Ok::<(), DimError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Keep(pub usize);

/// The axis argument of [`Ndarr::reduce`](crate::Ndarr::reduce) and
/// [`Ndarr::fold_axis`](crate::Ndarr::fold_axis): a `usize` removes the axis
/// (output rank [`Removed<R>`]); [`Keep`] keeps it with extent 1 (rank `R`).
/// Sealed; generic callers may use this bound.
pub trait ReduceAxis<R: Rank>: axis_count_sealed::Sealed {
    type Output: Rank;
    /// Whether the reduced axis stays with extent 1.
    const KEEP: bool;
    fn axis(self) -> usize;
}

impl<R: RemoveAxisRank> ReduceAxis<R> for usize {
    type Output = Removed<R>;
    const KEEP: bool = false;
    fn axis(self) -> usize {
        self
    }
}

impl<R: Rank> ReduceAxis<R> for Keep {
    type Output = R;
    const KEEP: bool = true;
    fn axis(self) -> usize {
        self.0
    }
}

/// A contraction count: `k` pairs trailing `self` axes with leading `other`
/// axes; `(s, k)` first shares `s` leading axes (a batch both operands walk
/// together), then contracts `k`. Typenum values preserve fixed output ranks;
/// `usize` counts select `Dyn`. Sealed; generic callers may use this bound.
pub trait ContractAxes<L: Rank, R: Rank>: axis_count_sealed::Sealed {
    type Output: Rank;
    /// Contracted axis pairs.
    fn count(self) -> usize;
    /// Shared leading axes.
    fn shared(&self) -> usize {
        0
    }
}

impl<L: Rank, R: Rank> ContractAxes<L, R> for (usize, usize) {
    type Output = Dyn;
    fn count(self) -> usize {
        self.1
    }
    fn shared(&self) -> usize {
        self.0
    }
}

impl<L, R, S, K> ContractAxes<L, R> for (S, K)
where
    L: ContractRank<R, K, S>,
    R: Rank,
    S: axis_count_sealed::Typenum,
    K: axis_count_sealed::Typenum,
{
    type Output = <L as ContractRank<R, K, S>>::Output;
    fn count(self) -> usize {
        K::USIZE
    }
    fn shared(&self) -> usize {
        S::USIZE
    }
}

impl<L: Rank, R: Rank> ContractAxes<L, R> for usize {
    type Output = Dyn;
    fn count(self) -> usize {
        self
    }
}

impl<L, R> ContractAxes<L, R> for typenum::UTerm
where
    L: ContractRank<R, Self>,
    R: Rank,
{
    type Output = <L as ContractRank<R, Self>>::Output;
    fn count(self) -> usize {
        0
    }
}

impl<L, R, U, B> ContractAxes<L, R> for typenum::UInt<U, B>
where
    L: ContractRank<R, Self>,
    R: Rank,
    Self: Unsigned,
{
    type Output = <L as ContractRank<R, Self>>::Output;
    fn count(self) -> usize {
        Self::USIZE
    }
}

/// Output rank selected by the operand ranks and contraction count type.
pub type Contracted<L, R, K> = <K as ContractAxes<L, R>>::Output;

/// Rank of the frame left after taking `K` trailing axes as cells. Fixed ranks
/// check `K` at compile time; `Dyn` stays `Dyn`.
#[diagnostic::on_unimplemented(
    message = "cannot take `{K}` trailing axes of rank `{Self}`",
    label = "the cell count must not exceed a fixed rank"
)]
pub trait FrameRank<K: Unsigned>: Rank {
    type Output: Rank;
}

impl<R, K> FrameRank<K> for R
where
    R: StaticRank + Sub<K, Output: StaticRank>,
    K: Unsigned + typenum::IsLessOrEqual<R, Output = typenum::True>,
{
    type Output = typenum::Diff<R, K>;
}

impl<K: Unsigned> FrameRank<K> for Dyn {
    type Output = Dyn;
}

/// A cell count for [`Ndarr::map_cells`](crate::Ndarr::map_cells): typenum
/// counts keep the cell rank and a fixed frame rank; `usize` counts select
/// `Dyn` for both. Sealed; generic callers may use this bound.
pub trait CellAxes<R: Rank>: axis_count_sealed::Sealed {
    type Cell: Rank;
    type Frame: Rank;
    fn count(self) -> usize;
}

impl<R: Rank> CellAxes<R> for usize {
    type Cell = Dyn;
    type Frame = Dyn;
    fn count(self) -> usize {
        self
    }
}

impl<R: FrameRank<Self>> CellAxes<R> for typenum::UTerm {
    type Cell = Self;
    type Frame = <R as FrameRank<Self>>::Output;
    fn count(self) -> usize {
        0
    }
}

impl<R, U, B> CellAxes<R> for typenum::UInt<U, B>
where
    R: FrameRank<Self>,
    Self: Rank + Unsigned,
{
    type Cell = Self;
    type Frame = <R as FrameRank<Self>>::Output;
    fn count(self) -> usize {
        Self::USIZE
    }
}

pub struct Dim<R: Rank> {
    shape: R::Store<usize>,
    rank: PhantomData<R>,
}

// Manual impl: the derive would demand `R: Debug` through the `PhantomData`.
impl<R: Rank> Debug for Dim<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Dim").field("shape", &self.shape).finish()
    }
}

impl<R: Rank> Clone for Dim<R> {
    fn clone(&self) -> Self {
        Self {
            shape: self.shape.clone(),
            rank: PhantomData,
        }
    }
}

impl<R: Rank> PartialEq for Dim<R> {
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl<R: Rank> Eq for Dim<R> {}

impl<R: Rank> Dim<R> {
    pub fn new(shape: &[usize]) -> Result<Self, DimError> {
        if R::FIXED_RANK.is_some_and(|rank| rank != shape.len()) {
            return Err(DimError::new(&format!(
                "Error initializing Dim of rank {:?} with slice of length {}",
                R::FIXED_RANK,
                shape.len()
            )));
        }
        let shape = R::Store::<usize>::try_from_slice(shape)
            .ok_or_else(|| DimError::new("Rank does not match shape length"))?;
        Ok(Self {
            shape,
            rank: PhantomData,
        })
    }

    pub fn as_slice(&self) -> &[usize] {
        self.shape.as_slice()
    }

    /// Extents of a fixed rank as an array, for destructuring. The pattern
    /// length must equal the rank at compile time; [`Dyn`] has no fixed length,
    /// so it keeps `as_slice`.
    ///
    /// ```
    /// use rapl::*;
    /// let x = Ndarr::<f32, U3>::zeros([2, 5, 8]);
    /// let [batch, time, model] = x.dim().to_array();
    /// assert_eq!((batch, time, model), (2, 5, 8));
    /// ```
    /// ```compile_fail
    /// let x = rapl::Ndarr::<f32, rapl::U3>::zeros([2, 5, 8]);
    /// let [rows, cols] = x.dim().to_array();
    /// ```
    pub fn to_array<const N: usize>(&self) -> [usize; N]
    where
        Const<N>: IntoArrayLength<ArrayLength = R>,
    {
        self.as_slice()
            .try_into()
            .expect("a fixed rank has exactly N extents")
    }

    /// Number of axes. Rank 0 is a scalar, not an empty collection, so there
    /// is no `is_empty`.
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self.as_slice().len()
    }

    pub fn get_number_elements(&self) -> usize {
        self.as_slice().iter().product()
    }

    pub fn remove_element(&self, index: usize) -> Dim<Removed<R>>
    where
        R: RemoveAxisRank,
    {
        assert!(index < self.len());
        let mut shape = self.as_slice().to_vec();
        shape.remove(index);
        Dim::new(&shape).expect("removing one axis matches the Removed rank")
    }

    pub fn insert_element(&self, index: usize, element: usize) -> Dim<Inserted<R>>
    where
        R: InsertAxisRank,
    {
        assert!(index <= self.len());
        let mut shape = self.as_slice().to_vec();
        shape.insert(index, element);
        Dim::new(&shape).expect("inserting one axis matches the Inserted rank")
    }

    pub fn broadcast_shape<R2: Rank>(
        &self,
        other: &Dim<R2>,
    ) -> Result<Dim<Broadcasted<R, R2>>, DimError>
    where
        R: BroadcastRank<R2>,
    {
        let left = self.as_slice();
        let right = other.as_slice();
        let len = left.len().max(right.len());
        let mut out = vec![0; len];
        for i in 0..len {
            let a = left
                .get(left.len().wrapping_sub(i + 1))
                .copied()
                .unwrap_or(1);
            let b = right
                .get(right.len().wrapping_sub(i + 1))
                .copied()
                .unwrap_or(1);
            if a != 1 && b != 1 && a != b {
                return Err(DimError::new(&format!(
                    "Error arrays with shape {:?} and {:?} can not be broadcasted.",
                    left, right
                )));
            }
            // Not `a.max(b)`: a zero-length axis paired with a unit axis stays zero.
            out[len - i - 1] = if a == 1 { b } else { a };
        }
        Ok(Dim::new(&out).expect("the longer rank is the Broadcasted rank"))
    }

    pub fn into_dyn(self) -> Dim<Dyn> {
        Dim::new(self.as_slice()).expect("Dyn accepts every rank")
    }

    pub fn into_ranked<R2: StaticRank>(self) -> Result<Dim<R2>, DimError> {
        Dim::new(self.as_slice())
    }
}

impl<R: Rank> From<&Dim<R>> for Dim<R> {
    fn from(value: &Dim<R>) -> Self {
        value.clone()
    }
}

impl<const N: usize> From<[usize; N]> for Dim<ConstArrayLength<N>>
where
    Const<N>: IntoArrayLength,
{
    fn from(value: [usize; N]) -> Self {
        Self::new(&value).expect("const array length determines rank")
    }
}

impl<const N: usize> From<&[usize; N]> for Dim<ConstArrayLength<N>>
where
    Const<N>: IntoArrayLength,
{
    fn from(value: &[usize; N]) -> Self {
        Self::new(value).expect("const array length determines rank")
    }
}

impl From<Vec<usize>> for Dim<Dyn> {
    fn from(value: Vec<usize>) -> Self {
        Self::new(&value).expect("Dyn accepts every rank")
    }
}

impl From<&[usize]> for Dim<Dyn> {
    fn from(value: &[usize]) -> Self {
        Self::new(value).expect("Dyn accepts every rank")
    }
}

impl From<usize> for Dim<U1> {
    fn from(value: usize) -> Self {
        Self::new(&[value]).expect("one extent has rank one")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use typenum::{U0, U2, U24, U3, U4, U6};

    #[test]
    fn fixed_and_dynamic_rank_are_distinct() {
        assert!(Dim::<U3>::new(&[1, 2, 4]).is_ok());
        assert!(Dim::<U3>::new(&[1, 2]).is_err());
        assert!(Dim::<U24>::new(&[2; 24]).is_ok());
        assert!(Dim::<U0>::new(&[]).is_ok());
        assert!(Dim::<U0>::new(&[1]).is_err());
        assert!(Dim::<Dyn>::new(&[1, 2, 4]).is_ok());
    }

    #[test]
    fn rank_relations() {
        let shape = Dim::<U3>::new(&[1, 2, 3]).unwrap();
        assert_eq!(shape.remove_element(1), Dim::<U2>::new(&[1, 3]).unwrap());
        assert_eq!(
            shape.remove_element(1).insert_element(1, 2),
            Dim::<U3>::new(&[1, 2, 3]).unwrap()
        );
    }

    fn broadcast(left: &[usize], right: &[usize]) -> Result<Vec<usize>, DimError> {
        let left = Dim::<Dyn>::new(left)?;
        let right = Dim::<Dyn>::new(right)?;
        Ok(left.broadcast_shape(&right)?.as_slice().to_vec())
    }

    /// Cases taken from the NumPy broadcasting rules and `np.broadcast_shapes`.
    #[test]
    fn numpy_broadcast_table() {
        let cases: &[(&[usize], &[usize], &[usize])] = &[
            (&[256, 256, 3], &[3], &[256, 256, 3]),
            (&[8, 1, 6, 1], &[7, 1, 5], &[8, 7, 6, 5]),
            (&[5, 4], &[1], &[5, 4]),
            (&[5, 4], &[4], &[5, 4]),
            (&[15, 3, 5], &[15, 1, 5], &[15, 3, 5]),
            (&[15, 3, 5], &[3, 5], &[15, 3, 5]),
            (&[15, 3, 5], &[3, 1], &[15, 3, 5]),
            // Scalars (rank 0) broadcast against anything.
            (&[], &[], &[]),
            (&[], &[3, 4], &[3, 4]),
            (&[2, 3], &[], &[2, 3]),
            (&[4, 1], &[3], &[4, 3]),
            (&[1, 3], &[4, 1], &[4, 3]),
            // Degenerate axes: a zero-length axis absorbs a unit axis.
            (&[0], &[1], &[0]),
            (&[0, 5], &[1, 5], &[0, 5]),
            (&[3, 0, 2], &[1, 1, 2], &[3, 0, 2]),
            (&[0], &[], &[0]),
            // Unit axes never win against a concrete extent.
            (&[1, 1, 1], &[2, 3, 4], &[2, 3, 4]),
            (&[6, 1, 1], &[1, 5, 4], &[6, 5, 4]),
        ];
        for (left, right, expected) in cases {
            assert_eq!(
                broadcast(left, right).expect("shapes conform"),
                *expected,
                "broadcasting {left:?} with {right:?}"
            );
            assert_eq!(
                broadcast(right, left).expect("shapes conform"),
                *expected,
                "broadcasting is commutative for {left:?} and {right:?}"
            );
        }
    }

    #[test]
    fn incompatible_shapes_are_rejected() {
        let cases: &[(&[usize], &[usize])] = &[
            (&[3], &[4]),
            (&[2, 1], &[8, 4, 3]),
            (&[1, 3, 4], &[2, 5, 4]),
            (&[256, 256, 3], &[4]),
            // A zero-length axis only conforms with itself or a unit axis.
            (&[0], &[3]),
            (&[3, 0], &[3, 2]),
        ];
        for (left, right) in cases {
            assert!(
                broadcast(left, right).is_err(),
                "{left:?} and {right:?} must not broadcast"
            );
            assert!(
                broadcast(right, left).is_err(),
                "{right:?} and {left:?} must not broadcast"
            );
        }
    }

    /// APL conformability: scalars and single-element axes extend, and rank is
    /// aligned from the trailing axis outward.
    #[test]
    fn apl_scalar_extension_and_conformability() {
        assert_eq!(broadcast(&[], &[2, 2]).unwrap(), vec![2, 2]);
        assert_eq!(broadcast(&[1], &[2, 3, 4]).unwrap(), vec![2, 3, 4]);
        assert_eq!(broadcast(&[2, 3], &[2, 3]).unwrap(), vec![2, 3]);
        // Leading-axis alignment is not APL-conformable here; NumPy pads on the
        // left, so a rank-1 argument matches the trailing axis only.
        assert_eq!(broadcast(&[2, 3], &[3]).unwrap(), vec![2, 3]);
        assert!(broadcast(&[2, 3], &[2]).is_err());
    }

    #[test]
    fn broadcast_is_associative_across_three_shapes() {
        let a = Dim::<Dyn>::new(&[8, 1, 6, 1]).unwrap();
        let b = Dim::<Dyn>::new(&[7, 1, 5]).unwrap();
        let c = Dim::<Dyn>::new(&[6, 5]).unwrap();

        let left = a.broadcast_shape(&b).unwrap().broadcast_shape(&c).unwrap();
        let right = b.broadcast_shape(&c).unwrap().broadcast_shape(&a).unwrap();

        assert_eq!(left.as_slice(), &[8, 7, 6, 5]);
        assert_eq!(left, right);
    }

    #[test]
    fn broadcast_of_static_ranks_keeps_the_larger_rank() {
        let matrix = Dim::<U2>::new(&[5, 4]).unwrap();
        let row = Dim::<U1>::new(&[4]).unwrap();
        let broadcasted: Dim<U2> = matrix.broadcast_shape(&row).unwrap();
        assert_eq!(broadcasted.as_slice(), &[5, 4]);

        let wide = Dim::<U4>::new(&[8, 1, 6, 1]).unwrap();
        let narrow = Dim::<U3>::new(&[7, 1, 5]).unwrap();
        let broadcasted: Dim<U4> = wide.broadcast_shape(&narrow).unwrap();
        assert_eq!(broadcasted.as_slice(), &[8, 7, 6, 5]);

        let scalar = Dim::<U0>::new(&[]).unwrap();
        let cube: Dim<U3> = scalar
            .broadcast_shape(&Dim::<U3>::new(&[2, 3, 4]).unwrap())
            .unwrap();
        assert_eq!(cube.as_slice(), &[2, 3, 4]);
    }

    #[test]
    fn broadcast_between_static_and_dynamic_rank_is_dynamic() {
        let fixed = Dim::<U3>::new(&[15, 1, 5]).unwrap();
        let dynamic = Dim::<Dyn>::new(&[3, 5]).unwrap();
        let broadcasted: Dim<Dyn> = fixed.broadcast_shape(&dynamic).unwrap();
        assert_eq!(broadcasted.as_slice(), &[15, 3, 5]);

        let broadcasted: Dim<Dyn> = dynamic.broadcast_shape(&fixed).unwrap();
        assert_eq!(broadcasted.as_slice(), &[15, 3, 5]);
    }

    #[test]
    fn broadcast_result_element_count_matches_the_shape() {
        let broadcasted = Dim::<U6>::new(&[1, 2, 1, 4, 1, 6])
            .unwrap()
            .broadcast_shape(&Dim::<U6>::new(&[1, 1, 3, 1, 5, 1]).unwrap())
            .unwrap();
        assert_eq!(broadcasted.as_slice(), &[1, 2, 3, 4, 5, 6]);
        assert_eq!(broadcasted.get_number_elements(), 720);

        let empty = Dim::<U2>::new(&[0, 4])
            .unwrap()
            .broadcast_shape(&Dim::<U2>::new(&[1, 4]).unwrap())
            .unwrap();
        assert_eq!(empty.get_number_elements(), 0);
    }
}
