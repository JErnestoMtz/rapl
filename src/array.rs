//! Generic N-dimensional arrays with rank-selected shape and stride metadata.

use crate::buffer::{Buffer, BufferMut};
use crate::errors::DimError;
use crate::shape::{Dim, InsertAxisRank, Inserted, Rank, RankStore, RemoveAxisRank, Removed};
use std::fmt::Debug;
use std::marker::PhantomData;
use std::ops::{Range, RangeFrom, RangeFull, RangeTo};
use typenum::{Unsigned, U0, U1, U2};

/// Contiguous C-order strides for a shape, in elements.
pub fn contiguous_strides(shape: &[usize]) -> Vec<isize> {
    let mut strides = vec![1isize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1] as isize;
    }
    strides
}

pub(crate) fn rank_strides<R: Rank>(strides: &[isize]) -> R::Store<isize> {
    R::Store::<isize>::try_from_slice(strides).expect("stride count must match rank")
}

/// Flat buffer offset for a multi-index given offset and strides.
pub fn flat_offset(offset: usize, strides: &[isize], indexes: &[usize]) -> usize {
    debug_assert_eq!(strides.len(), indexes.len());
    strides
        .iter()
        .zip(indexes)
        .fold(offset as isize, |position, (&stride, &index)| {
            position + stride * index as isize
        }) as usize
}

/// True when layout is contiguous C-order, independent of offset. Extent-1
/// axes carry no layout information, so their strides are ignored.
pub fn is_contiguous(shape: &[usize], strides: &[isize]) -> bool {
    if shape.contains(&0) {
        return true;
    }
    let mut expected = 1isize;
    for (&extent, &stride) in shape.iter().zip(strides).rev() {
        if extent != 1 {
            if stride != expected {
                return false;
            }
            expected *= extent as isize;
        }
    }
    true
}

mod index_int_sealed {
    pub trait Sealed {}
}

/// A primitive integer usable as an index, bound, or step in [`s!`](crate::s).
/// Every type is accepted, so integer literals still infer (they fall back to
/// `i32`) and `usize` extents need no casts. Values beyond `isize` saturate,
/// which selects exactly what the out-of-range value would.
pub trait IndexInt: Copy + index_int_sealed::Sealed {
    /// The value as an `isize`, saturating; never `isize::MIN`, so it negates.
    fn saturate(self) -> isize;
}

macro_rules! index_int {
    ($($int:ty),*) => {$(
        impl index_int_sealed::Sealed for $int {}
        impl IndexInt for $int {
            fn saturate(self) -> isize {
                (self as i128).clamp(isize::MIN as i128 + 1, isize::MAX as i128) as isize
            }
        }
    )*};
}

index_int!(i8, i16, i32, i64, isize, u8, u16, u32, u64, usize);

/// One rank-preserving axis selection: `start..end` taken every `step`
/// elements, normalized like NumPy (negative values count from the end, and
/// out-of-range bounds clamp). In [`s!`](crate::s) it is written `lo..hi;step`,
/// `..;-1`, and so on; ranges convert with `Slice::from`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Slice {
    pub start: Option<isize>,
    pub end: Option<isize>,
    pub step: isize,
}

impl Slice {
    /// `range` taken every `step` elements: the value of `range;step` in `s!`.
    pub fn stepped(range: impl Into<Slice>, step: impl IndexInt) -> Self {
        Self {
            step: step.saturate(),
            ..range.into()
        }
    }
}

impl From<RangeFull> for Slice {
    fn from(_: RangeFull) -> Self {
        Self {
            start: None,
            end: None,
            step: 1,
        }
    }
}

impl<I: IndexInt> From<Range<I>> for Slice {
    fn from(value: Range<I>) -> Self {
        Self {
            start: Some(value.start.saturate()),
            end: Some(value.end.saturate()),
            step: 1,
        }
    }
}

impl<I: IndexInt> From<RangeFrom<I>> for Slice {
    fn from(value: RangeFrom<I>) -> Self {
        Self {
            start: Some(value.start.saturate()),
            ..Self::from(..)
        }
    }
}

impl<I: IndexInt> From<RangeTo<I>> for Slice {
    fn from(value: RangeTo<I>) -> Self {
        Self {
            end: Some(value.end.saturate()),
            ..Self::from(..)
        }
    }
}

/// Inserts a length-one axis in [`Ndarr::slice`]; consumes no input axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NewAxis;

/// Sliding windows of extent `w` in [`Ndarr::slice`]: one input axis
/// (extent `n`, stride `s`) becomes a position axis (`n - w + 1`, stride `s`)
/// then a window axis (`w`, stride `s`), so `windows[p, k] == input[p + k]`
/// — an O(1) affine view, no copy. Windows alias one element into many
/// windows, so [`Ndarr::slice_mut`] rejects `Win`.
///
/// ```
/// use rapl::{s, Ndarr, Win, U2};
/// // Boxcar smoothing: mean over a width-3 window along axis 0.
/// let series = Ndarr::from([1.0, 2.0, 3.0, 4.0, 5.0]);
/// let smooth: Ndarr<f64, rapl::U1> = series
///     .slice(s![Win(3)]).unwrap()                  // [3, 3], zero copy
///     .fold_axis(1, 0.0, |acc, x| acc + x).unwrap()
///     .map(|sum| sum / 3.0);
/// assert_eq!(smooth, Ndarr::from([2.0, 3.0, 4.0]));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Win(pub usize);

/// One runtime selector of the basic-indexing grammar.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SliceSpec {
    /// Rank-preserving range with step.
    Slice(Slice),
    /// Scalar index: removes the axis. Negative wraps from the end.
    Index(isize),
    /// Insert a length-one axis; consumes no input axis.
    NewAxis,
    /// Sliding windows: the axis becomes positions then window (see [`Win`]).
    Win(usize),
}

impl From<Slice> for SliceSpec {
    fn from(value: Slice) -> Self {
        Self::Slice(value)
    }
}

impl From<RangeFull> for SliceSpec {
    fn from(value: RangeFull) -> Self {
        Self::Slice(value.into())
    }
}

impl<I: IndexInt> From<Range<I>> for SliceSpec {
    fn from(value: Range<I>) -> Self {
        Self::Slice(value.into())
    }
}

impl<I: IndexInt> From<RangeFrom<I>> for SliceSpec {
    fn from(value: RangeFrom<I>) -> Self {
        Self::Slice(value.into())
    }
}

impl<I: IndexInt> From<RangeTo<I>> for SliceSpec {
    fn from(value: RangeTo<I>) -> Self {
        Self::Slice(value.into())
    }
}

impl<I: IndexInt> From<I> for SliceSpec {
    fn from(value: I) -> Self {
        Self::Index(value.saturate())
    }
}

impl From<NewAxis> for SliceSpec {
    fn from(_: NewAxis) -> Self {
        Self::NewAxis
    }
}

impl From<Win> for SliceSpec {
    fn from(value: Win) -> Self {
        Self::Win(value.0)
    }
}

/// One typed selector of the basic-indexing grammar. `Out` is the number of
/// output axes it produces: `U0` for a scalar index, `U2` for [`Win`], `U1`
/// otherwise.
pub trait Selector: Clone {
    type Out: Unsigned;
    fn spec(self) -> SliceSpec;
}

macro_rules! impl_selector {
    (<$generic:ident: $bound:path> $type:ty, $out:ty) => {
        impl<$generic: $bound> Selector for $type {
            type Out = $out;
            fn spec(self) -> SliceSpec {
                SliceSpec::from(self)
            }
        }
    };
    ($type:ty, $out:ty) => {
        impl Selector for $type {
            type Out = $out;
            fn spec(self) -> SliceSpec {
                SliceSpec::from(self)
            }
        }
    };
}

impl_selector!(RangeFull, U1);
impl_selector!(<I: IndexInt> Range<I>, U1);
impl_selector!(<I: IndexInt> RangeFrom<I>, U1);
impl_selector!(<I: IndexInt> RangeTo<I>, U1);
impl_selector!(Slice, U1);
impl_selector!(NewAxis, U1);
impl_selector!(<I: IndexInt> I, U0);
impl_selector!(Win, U2);

/// Fold step for the output rank of a selection: `U0` leaves the rank alone,
/// `U1` applies [`InsertAxisRank`] once, `U2` twice (a [`Win`] selector
/// contributes two output axes).
pub trait BumpRank<C: Unsigned>: Rank {
    type Out: Rank;
}

impl<R: Rank> BumpRank<U0> for R {
    type Out = R;
}

impl<R: InsertAxisRank> BumpRank<U1> for R {
    type Out = Inserted<R>;
}

impl<R: InsertAxisRank<Output: InsertAxisRank>> BumpRank<U2> for R {
    type Out = Inserted<Inserted<R>>;
}

/// A full selection for [`Ndarr::slice`] / [`Ndarr::slice_mut`]: computes the
/// output rank and lowers to runtime [`SliceSpec`]s. Implemented for tuples of
/// [`Selector`]s (built by [`s!`](crate::s)) with compile-time output rank, and
/// for `&[SliceSpec]` as the runtime escape hatch (output rank [`Dyn`](crate::Dyn)).
pub trait SliceArgs {
    type OutRank: Rank;
    fn fill(&self, out: &mut Vec<SliceSpec>);
}

impl SliceArgs for &[SliceSpec] {
    type OutRank = crate::Dyn;
    fn fill(&self, out: &mut Vec<SliceSpec>) {
        out.extend_from_slice(self);
    }
}

macro_rules! impl_slice_args {
    () => {
        impl SliceArgs for () {
            type OutRank = U0;
            fn fill(&self, _out: &mut Vec<SliceSpec>) {}
        }
    };
    ($head:ident $(, $tail:ident)*) => {
        #[allow(non_snake_case)]
        impl<$head: Selector $(, $tail: Selector)*> SliceArgs for ($head, $($tail,)*)
        where
            ($($tail,)*): SliceArgs,
            <($($tail,)*) as SliceArgs>::OutRank: BumpRank<$head::Out>,
        {
            type OutRank =
                <<($($tail,)*) as SliceArgs>::OutRank as BumpRank<$head::Out>>::Out;
            fn fill(&self, out: &mut Vec<SliceSpec>) {
                let ($head, $($tail,)*) = self.clone();
                out.push($head.spec());
                ($($tail,)*).fill(out);
            }
        }
        impl_slice_args!($($tail),*);
    };
}

impl_slice_args!(S0, S1, S2, S3, S4, S5, S6, S7);

/// Build a basic-indexing selection: every input axis is named by exactly one
/// non-[`NewAxis`] selector. An integer removes its axis, [`NewAxis`] inserts
/// a length-one axis, ranges and [`Slice`] keep theirs, and [`Win`] turns its
/// axis into two: positions then window. `range;step` takes every `step`-th
/// element of a range (`..;-1` reverses). Indices, bounds, and steps may be any
/// primitive integer type.
///
/// With a negative step, bounds run high to low as in NumPy: `3..1;-1` selects
/// 3 and 2. Clippy's `reversed_empty_ranges` rejects such literal ranges; use
/// `-1..1;-1`, variables, or a [`Slice`] literal.
///
/// ```
/// use rapl::{s, Ndarr, NewAxis, Win, U1, U3};
/// let a = Ndarr::from([[1, 2, 3], [4, 5, 6]]);
/// let row: rapl::NdView<'_, i32, U1> = a.slice(s![1, 1..]).unwrap();
/// assert_eq!(row.iter_elems().copied().collect::<Vec<_>>(), vec![5, 6]);
/// let up: rapl::NdView<'_, i32, U3> = a.slice(s![NewAxis, .., ..;-1]).unwrap();
/// assert_eq!(up.shape(), &[1, 2, 3]);
/// let (first, stride) = (0usize, 2usize);
/// let odd: rapl::NdView<'_, i32, rapl::U2> = a.slice(s![.., first..;stride]).unwrap();
/// assert_eq!(odd, Ndarr::from([[1, 3], [4, 6]]));
/// let windows: rapl::NdView<'_, i32, U3> = a.slice(s![.., Win(2)]).unwrap();
/// assert_eq!(windows.shape(), &[2, 2, 2]);
/// ```
#[macro_export]
macro_rules! s {
    (@selector $selector:expr) => {
        $selector
    };
    (@selector $range:expr; $step:expr) => {
        $crate::Slice::stepped($range, $step)
    };
    ($($selector:expr $(; $step:expr)?),* $(,)?) => {
        ($($crate::s!(@selector $selector $(; $step)?),)*)
    };
}

/// N-dimensional array of `T` with rank `R`. Values live in `buffer`; the
/// offset, shape, and strides describe the view onto it.
///
/// `B` selects ownership: `Vec<T>` (the default) owns its elements, `&[T]` and
/// `&mut [T]` borrow them (see [`NdView`] and [`NdViewMut`]).
///
/// View adapters (`slice`, `t_view`, ...) chain without intermediate bindings.
/// `permute_axes`, `reshape` and `into_dyn` consume `self` and keep its
/// storage; call `view()` first to keep the source.
///
/// Index with one coordinate per axis, `a[[i, j]]`, or a runtime `&[usize]`.
/// A fixed rank rejects the wrong coordinate count at build time (not during
/// `cargo check`); [`Dyn`](crate::Dyn) panics at runtime.
///
/// ```
/// use rapl::*;
/// let fixed = Ndarr::from([[1, 2], [3, 4]]);
/// let dynamic = fixed.clone().into_dyn();
/// assert_eq!(fixed[[1, 0]], dynamic[[1, 0]]);
/// ```
/// ```compile_fail,E0080
/// let fixed = rapl::Ndarr::from([[1, 2], [3, 4]]);
/// let _ = fixed[[1, 0, 0]];
/// ```
pub struct Ndarr<T, R: Rank, B: Buffer<T> = Vec<T>> {
    pub(crate) buffer: B,
    pub(crate) offset: usize,
    pub(crate) dim: Dim<R>,
    pub(crate) strides: R::Store<isize>,
    pub(crate) elem: PhantomData<T>,
}

/// Borrowed, read-only view: an [`Ndarr`] over `&[T]`.
pub type NdView<'a, T, R> = Ndarr<T, R, &'a [T]>;
/// Borrowed, mutable view: an [`Ndarr`] over `&mut [T]`.
pub type NdViewMut<'a, T, R> = Ndarr<T, R, &'a mut [T]>;

/// Assemble an array from runtime metadata, checking the shape fits rank `R`.
/// Every view constructor lowers to this.
fn assemble<T, R: Rank, B: Buffer<T>>(
    buffer: B,
    offset: usize,
    shape: &[usize],
    strides: &[isize],
) -> Result<Ndarr<T, R, B>, DimError> {
    Ok(Ndarr {
        buffer,
        offset,
        dim: Dim::new(shape)?,
        strides: rank_strides::<R>(strides),
        elem: PhantomData,
    })
}

// `{:?}` shows data and dim, not the private offset/strides view metadata.
impl<T: Debug, R: Rank, B: Buffer<T>> Debug for Ndarr<T, R, B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Ndarr")
            .field("data", &self.iter_elems().collect::<Vec<_>>())
            .field("dim", &self.dim)
            .finish()
    }
}

impl<T, R: Rank, B: Buffer<T> + Clone> Clone for Ndarr<T, R, B> {
    fn clone(&self) -> Self {
        Self {
            buffer: self.buffer.clone(),
            offset: self.offset,
            dim: self.dim.clone(),
            strides: self.strides.clone(),
            elem: PhantomData,
        }
    }
}

fn normalize_slice(axis_len: usize, slice: Slice) -> Result<(isize, usize, isize), DimError> {
    if slice.step == 0 {
        return Err(DimError::new("Slice step cannot be zero"));
    }
    let len = isize::try_from(axis_len).map_err(|_| DimError::new("Axis is too large"))?;
    if slice.step > 0 {
        let normalize = |value: isize| {
            let value = if value < 0 {
                value.saturating_add(len)
            } else {
                value
            };
            value.clamp(0, len)
        };
        let start = slice.start.map(normalize).unwrap_or(0);
        let end = slice.end.map(normalize).unwrap_or(len);
        let count = if start >= end {
            0
        } else {
            usize::try_from((end - start - 1) / slice.step + 1)
                .map_err(|_| DimError::new("Slice length overflow"))?
        };
        Ok((start, count, slice.step))
    } else {
        let normalize = |value: isize| {
            let value = if value < 0 {
                value.saturating_add(len)
            } else {
                value
            };
            value.clamp(-1, len.saturating_sub(1))
        };
        let start = slice
            .start
            .map(normalize)
            .unwrap_or_else(|| len.saturating_sub(1));
        // `None` is the exclusive sentinel before axis zero; an explicit -1
        // means the last element, matching Python/NumPy slice normalization.
        let end = slice.end.map(normalize).unwrap_or(-1);
        let step = slice
            .step
            .checked_neg()
            .ok_or_else(|| DimError::new("Slice step overflow"))?;
        let count = if start <= end {
            0
        } else {
            usize::try_from((start - end - 1) / step + 1)
                .map_err(|_| DimError::new("Slice length overflow"))?
        };
        Ok((start, count, slice.step))
    }
}

/// The one lowering every selection shares: offset, shape, and strides of the
/// view described by `specs`. Every input axis must be named by exactly one
/// non-`NewAxis` spec.
fn selected_metadata(
    offset: usize,
    shape: &[usize],
    strides: &[isize],
    specs: &[SliceSpec],
) -> Result<(usize, Vec<usize>, Vec<isize>), DimError> {
    let consumed = specs
        .iter()
        .filter(|spec| !matches!(spec, SliceSpec::NewAxis))
        .count();
    if consumed != shape.len() {
        return Err(DimError::new("Selection must name every axis exactly once"));
    }
    let mut out_offset = isize::try_from(offset).map_err(|_| DimError::new("Offset overflow"))?;
    let mut out_shape = Vec::with_capacity(specs.len());
    let mut out_strides = Vec::with_capacity(specs.len());
    let mut axis = 0;

    for spec in specs {
        match *spec {
            SliceSpec::NewAxis => {
                out_shape.push(1);
                out_strides.push(0);
            }
            SliceSpec::Index(index) => {
                let len =
                    isize::try_from(shape[axis]).map_err(|_| DimError::new("Axis is too large"))?;
                let wrapped = if index < 0 { index + len } else { index };
                if wrapped < 0 || wrapped >= len {
                    return Err(DimError::new("Index out of bounds"));
                }
                let delta = strides[axis]
                    .checked_mul(wrapped)
                    .ok_or_else(|| DimError::new("Offset overflow"))?;
                out_offset = out_offset
                    .checked_add(delta)
                    .ok_or_else(|| DimError::new("Offset overflow"))?;
                axis += 1;
            }
            SliceSpec::Slice(slice) => {
                let (start, count, step) = normalize_slice(shape[axis], slice)?;
                if count != 0 {
                    let delta = strides[axis]
                        .checked_mul(start)
                        .ok_or_else(|| DimError::new("Slice offset overflow"))?;
                    out_offset = out_offset
                        .checked_add(delta)
                        .ok_or_else(|| DimError::new("Slice offset overflow"))?;
                }
                out_shape.push(count);
                out_strides.push(
                    strides[axis]
                        .checked_mul(step)
                        .ok_or_else(|| DimError::new("Slice stride overflow"))?,
                );
                axis += 1;
            }
            SliceSpec::Win(window) => {
                let extent = shape[axis];
                if window > extent {
                    return Err(DimError::new("Window is longer than its axis"));
                }
                // Both output axes reuse the input stride: positions first,
                // then the window, so windows[p, k] == input[p + k].
                out_shape.push(extent - window + 1);
                out_strides.push(strides[axis]);
                out_shape.push(window);
                out_strides.push(strides[axis]);
                axis += 1;
            }
        }
    }

    let out_offset = usize::try_from(out_offset)
        .map_err(|_| DimError::new("Selection points before the buffer"))?;
    Ok((out_offset, out_shape, out_strides))
}

impl<T, R: Rank, B: Buffer<T>> Ndarr<T, R, B> {
    pub fn rank(&self) -> usize {
        self.dim.len()
    }

    pub fn shape(&self) -> &[usize] {
        self.dim.as_slice()
    }

    pub fn dim(&self) -> &Dim<R> {
        &self.dim
    }

    pub fn strides(&self) -> &[isize] {
        self.strides.as_slice()
    }

    pub fn offset(&self) -> usize {
        self.offset
    }

    pub fn len(&self) -> usize {
        self.dim.get_number_elements()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn is_standard_layout(&self) -> bool {
        is_contiguous(self.shape(), self.strides())
    }

    pub fn flat_index(&self, indexes: &[usize]) -> Result<usize, DimError> {
        if indexes.len() != self.rank() {
            return Err(DimError::new("Index rank does not match array rank"));
        }
        if indexes
            .iter()
            .zip(self.shape())
            .any(|(&index, &dimension)| index >= dimension)
        {
            return Err(DimError::new("Index out of bounds"));
        }
        Ok(flat_offset(self.offset, self.strides(), indexes))
    }

    /// Borrow elements in logical C-order; use `.cloned()` to materialize values.
    pub fn iter_elems(&self) -> ElemIter<'_, T> {
        ElemIter {
            buffer: self.buffer.as_slice(),
            positions: PosIter::new(self.shape(), self.strides(), self.offset, self.len()),
        }
    }

    /// One-dimensional views parallel to `axis`, in logical C-order over the
    /// non-lane axes, independent of the input's physical layout.
    pub fn lanes(
        &self,
        axis: usize,
    ) -> Result<impl ExactSizeIterator<Item = NdView<'_, T, U1>> + '_, DimError> {
        if axis >= self.rank() {
            return Err(DimError::new("Axis greater than rank"));
        }
        let mut frame = self.shape().to_vec();
        let mut frame_strides = self.strides().to_vec();
        let lane_dim = Dim::new(&[frame.remove(axis)]).expect("a lane has rank one");
        let lane_strides = rank_strides::<U1>(&[frame_strides.remove(axis)]);
        Ok(self.cells(&frame, &frame_strides, lane_dim, lane_strides))
    }

    /// The one cell walk: a borrowed view with metadata `dim`/`strides` at
    /// each position of the frame, in logical C-order. Lanes are one-axis
    /// cells; `map_cells` takes the trailing axes as the cell.
    pub(crate) fn cells<'a, C: Rank>(
        &'a self,
        frame: &[usize],
        frame_strides: &[isize],
        dim: Dim<C>,
        strides: C::Store<isize>,
    ) -> impl ExactSizeIterator<Item = NdView<'a, T, C>> {
        let buffer = self.buffer.as_slice();
        let count = frame.iter().product();
        PosIter::new(frame, frame_strides, self.offset, count).map(move |offset| Ndarr {
            buffer,
            offset,
            dim: dim.clone(),
            strides: strides.clone(),
            elem: PhantomData,
        })
    }

    /// Borrow the whole array as a read-only view.
    pub fn view(&self) -> Ndarr<T, R, B::Reborrow<'_>> {
        Ndarr {
            buffer: self.buffer.reborrow(),
            offset: self.offset,
            dim: self.dim.clone(),
            strides: self.strides.clone(),
            elem: PhantomData,
        }
    }

    /// Reorder axes in O(1), consuming `self` and keeping its storage: axis `i`
    /// of the result is axis `order[i]` of the source, so `order` must list
    /// every axis exactly once. Borrow with `a.view().permute_axes(order)`; an
    /// owned array stays owned, with its axes no longer in C order.
    ///
    /// ```
    /// use rapl::*;
    /// let a = Ndarr::from([[1, 2, 3], [4, 5, 6]]);
    /// assert_eq!(a.view().permute_axes(&[1, 0])?, Ndarr::from([[1, 4], [2, 5], [3, 6]]));
    /// let owned: Ndarr<i32, U2> = a.permute_axes(&[1, 0])?; // no copy
    /// assert!(!owned.is_standard_layout());
    /// assert_eq!(owned.into_data(), vec![1, 4, 2, 5, 3, 6]);
    /// # Ok::<(), DimError>(())
    /// ```
    pub fn permute_axes(self, order: &[usize]) -> Result<Self, DimError> {
        if order.len() != self.rank() {
            return Err(DimError::new(&format!(
                "A permutation of a rank {} array needs {} axes, {} were provided.",
                self.rank(),
                self.rank(),
                order.len()
            )));
        }
        let mut seen = vec![false; self.rank()];
        for &axis in order {
            if axis >= self.rank() {
                return Err(DimError::new("Axis greater than rank"));
            }
            if std::mem::replace(&mut seen[axis], true) {
                return Err(DimError::new("Permutation repeats an axis"));
            }
        }
        let shape: Vec<usize> = order.iter().map(|&axis| self.shape()[axis]).collect();
        let strides: Vec<isize> = order.iter().map(|&axis| self.strides()[axis]).collect();
        assemble(self.buffer, self.offset, &shape, &strides)
    }

    pub fn t_view(&self) -> Ndarr<T, R, B::Reborrow<'_>> {
        let order: Vec<usize> = (0..self.rank()).rev().collect();
        self.view()
            .permute_axes(&order)
            .expect("reversing the axis order is a permutation")
    }

    /// Transpose into an owned array.
    pub fn t(&self) -> Ndarr<T, R>
    where
        T: Clone,
    {
        self.t_view().to_owned_array()
    }

    /// Reshape in O(1), consuming `self` and keeping its storage: owned stays
    /// owned, a view stays a view (borrow with `a.view().reshape(shape)`).
    /// Requires standard C-order layout and never copies; materialize a
    /// sliced, permuted, or broadcast array first with `to_owned_array()`.
    pub fn reshape<R2: Rank, D: Into<Dim<R2>>>(
        self,
        shape: D,
    ) -> Result<Ndarr<T, R2, B>, DimError> {
        let shape = shape.into();
        if self.len() != shape.get_number_elements() {
            return Err(DimError::new(&format!(
                "Can not reshape array with shape {:?} to {:?}.",
                self.shape(),
                shape.as_slice()
            )));
        }
        if !self.is_standard_layout() {
            return Err(DimError::new(
                "reshape requires standard C-order layout; materialize with to_owned_array()",
            ));
        }
        Ok(Ndarr {
            buffer: self.buffer,
            offset: self.offset,
            strides: rank_strides::<R2>(&contiguous_strides(shape.as_slice())),
            dim: shape,
            elem: PhantomData,
        })
    }

    /// Runtime-axis form of `s![.., NewAxis, ..]`: insert a length-one axis
    /// at a position known only at runtime, without moving elements.
    pub fn insert_axis_view(
        &self,
        axis: usize,
    ) -> Result<Ndarr<T, Inserted<R>, B::Reborrow<'_>>, DimError>
    where
        R: InsertAxisRank,
    {
        if axis > self.rank() {
            return Err(DimError::new("Axis greater than rank"));
        }
        let mut specs: Vec<SliceSpec> = vec![SliceSpec::from(..); self.rank()];
        specs.insert(axis, SliceSpec::NewAxis);
        let (offset, shape, strides) =
            selected_metadata(self.offset, self.shape(), self.strides(), &specs)?;
        assemble(self.buffer.reborrow(), offset, &shape, &strides)
    }

    /// Runtime-axis form of `s![.., index, ..]`: select `index` along an axis
    /// known only at runtime, removing that axis.
    pub fn index_axis_view(
        &self,
        axis: usize,
        index: usize,
    ) -> Result<Ndarr<T, Removed<R>, B::Reborrow<'_>>, DimError>
    where
        R: RemoveAxisRank,
    {
        let (offset, shape, strides) = self.indexed_axis_metadata(axis, index)?;
        assemble(self.buffer.reborrow(), offset, &shape, &strides)
    }

    fn indexed_axis_metadata(
        &self,
        axis: usize,
        index: usize,
    ) -> Result<(usize, Vec<usize>, Vec<isize>), DimError> {
        if axis >= self.rank() {
            return Err(DimError::new("Axis greater than rank"));
        }
        let mut specs: Vec<SliceSpec> = vec![SliceSpec::from(..); self.rank()];
        specs[axis] =
            SliceSpec::Index(isize::try_from(index).map_err(|_| DimError::new("Index overflow"))?);
        selected_metadata(self.offset, self.shape(), self.strides(), &specs)
    }

    /// Broadcast to `new_dim` as an O(1) view (NumPy's `broadcast_to`):
    /// every source axis must equal its right-aligned target axis or be 1.
    /// Materialize with [`to_owned_array`](Self::to_owned_array).
    pub fn broadcast_view_to<ROut: Rank>(
        &self,
        new_dim: &Dim<ROut>,
    ) -> Result<Ndarr<T, ROut, B::Reborrow<'_>>, DimError> {
        if new_dim.len() < self.rank() {
            return Err(DimError::new("Array can not be broadcasted to shape"));
        }
        let mut strides = vec![0isize; new_dim.len()];
        for i in 0..self.rank() {
            let source = self.rank() - i - 1;
            let target = new_dim.len() - i - 1;
            let source_len = self.shape()[source];
            let target_len = new_dim.as_slice()[target];
            if source_len != target_len && source_len != 1 {
                return Err(DimError::new("Array can not be broadcasted to shape"));
            }
            if source_len == target_len {
                strides[target] = self.strides()[source];
            }
        }
        Ok(Ndarr {
            buffer: self.buffer.reborrow(),
            offset: self.offset,
            dim: new_dim.clone(),
            strides: rank_strides::<ROut>(&strides),
            elem: PhantomData,
        })
    }

    /// Basic-indexing selection: integer indexes remove axes, [`NewAxis`]
    /// inserts them, ranges and [`Slice`]s keep them, and [`Win`] doubles its
    /// axis into positions and window. Typed tuples from [`s!`](crate::s) compute the
    /// output rank at compile time; `&[SliceSpec]` is the runtime escape
    /// hatch and yields [`Dyn`](crate::Dyn).
    pub fn slice<A: SliceArgs>(
        &self,
        args: A,
    ) -> Result<Ndarr<T, A::OutRank, B::Reborrow<'_>>, DimError> {
        let mut specs = Vec::new();
        args.fill(&mut specs);
        let (offset, shape, strides) =
            selected_metadata(self.offset, self.shape(), self.strides(), &specs)?;
        assemble(self.buffer.reborrow(), offset, &shape, &strides)
    }
}

impl<T, R: Rank, B: BufferMut<T>> Ndarr<T, R, B> {
    pub fn view_mut(&mut self) -> NdViewMut<'_, T, R> {
        Ndarr {
            buffer: self.buffer.as_mut_slice(),
            offset: self.offset,
            dim: self.dim.clone(),
            strides: self.strides.clone(),
            elem: PhantomData,
        }
    }

    /// Mutable counterpart of [`Ndarr::index_axis_view`].
    pub fn index_axis_mut(
        &mut self,
        axis: usize,
        index: usize,
    ) -> Result<NdViewMut<'_, T, Removed<R>>, DimError>
    where
        R: RemoveAxisRank,
    {
        let (offset, shape, strides) = self.indexed_axis_metadata(axis, index)?;
        assemble(self.buffer.as_mut_slice(), offset, &shape, &strides)
    }

    /// Mutable counterpart of [`Ndarr::slice`]; [`Win`] is rejected because
    /// windows alias and mutable views must not.
    pub fn slice_mut<A: SliceArgs>(
        &mut self,
        args: A,
    ) -> Result<NdViewMut<'_, T, A::OutRank>, DimError> {
        let mut specs = Vec::new();
        args.fill(&mut specs);
        if specs.iter().any(|spec| matches!(spec, SliceSpec::Win(_))) {
            return Err(DimError::new("Windows alias; Win is read-only"));
        }
        let (offset, shape, strides) =
            selected_metadata(self.offset, self.shape(), self.strides(), &specs)?;
        assemble(self.buffer.as_mut_slice(), offset, &shape, &strides)
    }
}

/// Owned arrays hold every element of their `Vec` exactly once, at offset
/// zero. Constructors and computed results are in C order; `permute_axes` can
/// leave an owned array's axes permuted, which `is_standard_layout` reports.
impl<T, R: Rank> Ndarr<T, R> {
    /// Take ownership of `data` as a contiguous array of shape `dim`; element
    /// count must match.
    pub fn from_vec_dim(data: Vec<T>, dim: Dim<R>) -> Result<Self, DimError> {
        if data.len() != dim.get_number_elements() {
            return Err(DimError::new(&format!(
                "The number of elements of an Ndarray of shape {:?} is {}, and {} were provided.",
                dim.as_slice(),
                dim.get_number_elements(),
                data.len()
            )));
        }
        Ok(Self::contiguous(data, dim))
    }

    /// [`from_vec_dim`](Self::from_vec_dim) without the length check, for data
    /// whose length is correct by construction.
    pub(crate) fn contiguous(data: Vec<T>, dim: Dim<R>) -> Self {
        debug_assert_eq!(data.len(), dim.get_number_elements());
        let strides = rank_strides::<R>(&contiguous_strides(dim.as_slice()));
        Self {
            buffer: data,
            offset: 0,
            dim,
            strides,
            elem: PhantomData,
        }
    }

    /// Copy `data` into a new array of shape `shape`.
    pub fn new<D: Into<Dim<R>>>(data: &[T], shape: D) -> Result<Self, DimError>
    where
        T: Clone,
    {
        Self::from_vec_dim(data.to_vec(), shape.into())
    }

    pub fn from<P: Into<Self>>(p: P) -> Self {
        p.into()
    }

    /// Elements in logical C-order. Panics for an array whose axes were
    /// permuted; use `into_data` or `to_owned_array` there.
    pub fn data(&self) -> &[T] {
        self.assert_standard_layout();
        &self.buffer
    }

    /// Mutable elements in logical C-order; panics like [`data`](Self::data).
    pub fn data_mut(&mut self) -> &mut [T] {
        self.assert_standard_layout();
        &mut self.buffer
    }

    fn assert_standard_layout(&self) {
        assert!(
            self.is_standard_layout(),
            "raw data of a permuted array is not in C order; use into_data() or to_owned_array()"
        );
    }

    /// Elements in logical C-order. Moves the allocation when the layout is
    /// standard; a permuted array is reordered without cloning.
    pub fn into_data(self) -> Vec<T> {
        if self.is_standard_layout() {
            return self.buffer;
        }
        let positions = PosIter::new(self.shape(), self.strides(), self.offset, self.len());
        let mut slots: Vec<Option<T>> = self.buffer.into_iter().map(Some).collect();
        positions
            .map(|p| {
                slots[p]
                    .take()
                    .expect("an owned array visits each element once")
            })
            .collect()
    }

    /// Erase the static rank; the element allocation is moved unchanged.
    pub fn into_dyn(self) -> Ndarr<T, crate::Dyn> {
        let strides = rank_strides::<crate::Dyn>(self.strides());
        let dim = self.dim.into_dyn();
        Ndarr {
            buffer: self.buffer,
            offset: self.offset,
            strides,
            dim,
            elem: PhantomData,
        }
    }

    /// Check the runtime rank against `R2`; the allocation is moved unchanged.
    pub fn into_ranked<R2: crate::StaticRank>(self) -> Result<Ndarr<T, R2>, DimError> {
        let dim = self.dim.clone().into_ranked::<R2>()?;
        let strides = rank_strides::<R2>(self.strides());
        Ok(Ndarr {
            buffer: self.buffer,
            offset: self.offset,
            dim,
            strides,
            elem: PhantomData,
        })
    }
}

impl<T> Ndarr<T, U0> {
    /// The single element of a rank-0 array.
    pub fn scalar(self) -> T {
        self.buffer
            .into_iter()
            .next()
            .expect("a rank-0 array holds exactly one element")
    }
}

/// Strided odometer over an array's logical C-order buffer positions: keeps a
/// running position, adds the last axis's stride per step, and corrects on
/// carry. O(1) amortized per step. Owns its metadata so callers can walk
/// positions while mutating the buffer.
pub(crate) struct PosIter {
    shape: Vec<usize>,
    strides: Vec<isize>,
    idx: Vec<usize>,
    pos: isize,
    remaining: usize,
}

impl PosIter {
    pub(crate) fn new(shape: &[usize], strides: &[isize], offset: usize, len: usize) -> Self {
        PosIter {
            shape: shape.to_vec(),
            strides: strides.to_vec(),
            idx: vec![0; shape.len()],
            pos: offset as isize,
            remaining: len,
        }
    }

    /// Coordinates of the element the next step yields.
    pub(crate) fn coordinates(&self) -> &[usize] {
        &self.idx
    }
}

impl Iterator for PosIter {
    type Item = usize;

    fn next(&mut self) -> Option<usize> {
        if self.remaining == 0 {
            return None;
        }
        self.remaining -= 1;
        let current = self.pos as usize;
        for i in (0..self.shape.len()).rev() {
            self.idx[i] += 1;
            self.pos += self.strides[i];
            if self.idx[i] < self.shape[i] {
                break;
            }
            self.idx[i] = 0;
            self.pos -= self.shape[i] as isize * self.strides[i];
        }
        Some(current)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl ExactSizeIterator for PosIter {}

/// Borrowed elements of an array in logical C-order.
pub struct ElemIter<'a, T> {
    buffer: &'a [T],
    positions: PosIter,
}

impl<'a, T> Iterator for ElemIter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        self.positions.next().map(|p| &self.buffer[p])
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.positions.size_hint()
    }
}

impl<T1, T2, R1, R2, B1, B2> PartialEq<Ndarr<T2, R2, B2>> for Ndarr<T1, R1, B1>
where
    T1: PartialEq<T2>,
    R1: Rank,
    R2: Rank,
    B1: Buffer<T1>,
    B2: Buffer<T2>,
{
    fn eq(&self, other: &Ndarr<T2, R2, B2>) -> bool {
        self.shape() == other.shape()
            && self
                .iter_elems()
                .zip(other.iter_elems())
                .all(|(left, right)| left == right)
    }
}

impl<T: Eq, R: Rank, B: Buffer<T>> Eq for Ndarr<T, R, B> {}

#[cfg(test)]
mod tests {
    use super::*;
    use typenum::{U2, U3};

    #[test]
    fn owned_roundtrip() {
        let array =
            Ndarr::<i32, U2>::from_vec_dim(vec![1, 2, 3, 4], Dim::new(&[2, 2]).unwrap()).unwrap();
        assert_eq!(array[[1, 0]], 3);
        assert_eq!(array.flat_index(&[1, 0]).unwrap(), 2);
    }

    #[test]
    fn from_vec_dim_checks_the_element_count() {
        let short = Ndarr::<i32, U2>::from_vec_dim(vec![1, 2, 3], Dim::new(&[2, 2]).unwrap());
        assert!(short.is_err());
        assert!(Ndarr::<i32, U2>::new(&[1, 2, 3], [2, 2]).is_err());
    }

    #[test]
    fn slicing_normalizes_like_numpy() {
        let array = Ndarr::<i32, U2>::new(&(0..12).collect::<Vec<_>>(), [3, 4]).unwrap();
        let reversed = array.slice(s![.., ..;-1]).unwrap();
        assert_eq!(reversed.shape(), &[3, 4]);
        assert_eq!(
            reversed.iter_elems().cloned().collect::<Vec<_>>(),
            &[3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8]
        );
    }

    #[test]
    fn permutation_reorders_axes_without_copying() {
        let array = Ndarr::<i32, U3>::new(&(0..24).collect::<Vec<_>>(), [2, 3, 4]).unwrap();
        let permuted = array.view().permute_axes(&[2, 0, 1]).unwrap();
        assert_eq!(permuted.shape(), &[4, 2, 3]);
        for i in 0..2 {
            for j in 0..3 {
                for k in 0..4 {
                    assert_eq!(permuted[[k, i, j]], array[[i, j, k]]);
                }
            }
        }
    }

    #[test]
    fn axis_movement_is_permutation_order() {
        let array = Ndarr::<i32, U3>::new(&(0..24).collect::<Vec<_>>(), [2, 3, 4]).unwrap();
        assert_eq!(
            array.t_view(),
            array.view().permute_axes(&[2, 1, 0]).unwrap()
        );
        let swap_0_2 = array.view().permute_axes(&[2, 1, 0]).unwrap();
        assert_eq!(swap_0_2.shape(), &[4, 3, 2]);
        let move_0_to_2 = array.view().permute_axes(&[1, 2, 0]).unwrap();
        assert_eq!(move_0_to_2.shape(), &[3, 4, 2]);
    }

    #[test]
    fn permutation_composes_with_other_views() {
        let array = Ndarr::<i32, U3>::new(&(0..24).collect::<Vec<_>>(), [2, 3, 4]).unwrap();
        let sliced = array.slice(s![.., 1..3, ..;-1]).unwrap();
        let permuted = sliced.view().permute_axes(&[1, 2, 0]).unwrap();
        assert_eq!(permuted.shape(), &[2, 4, 2]);
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..4 {
                    assert_eq!(permuted[[j, k, i]], sliced[[i, j, k]]);
                }
            }
        }
        assert_eq!(permuted.t_view().t_view(), permuted);
    }

    #[test]
    fn permutation_rejects_malformed_orders() {
        let array = Ndarr::<i32, U3>::new(&(0..24).collect::<Vec<_>>(), [2, 3, 4]).unwrap();
        assert!(array.view().permute_axes(&[0, 1]).is_err());
        assert!(array.view().permute_axes(&[0, 1, 2, 0]).is_err());
        assert!(array.view().permute_axes(&[0, 1, 3]).is_err());
        assert!(array.view().permute_axes(&[0, 1, 1]).is_err());
    }

    #[test]
    fn permutation_handles_zero_and_unit_extents() {
        let empty = Ndarr::<i32, U3>::new(&[], [2, 0, 4]).unwrap();
        assert_eq!(
            empty.view().permute_axes(&[1, 2, 0]).unwrap().shape(),
            &[0, 4, 2]
        );
        let unit = Ndarr::<i32, U3>::new(&[7], [1, 1, 1]).unwrap();
        assert_eq!(
            unit.view().permute_axes(&[2, 0, 1]).unwrap().shape(),
            &[1, 1, 1]
        );
    }
}
