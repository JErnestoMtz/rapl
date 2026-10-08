use crate::array::{rank_strides, NdView, Ndarr, PosIter, SliceSpec};
use crate::buffer::{Buffer, BufferMut};
use crate::{CellAxes, Dim, DimError, Rank, ReduceAxis, U1};

mod dyadic;

/// Order in which [`Ndarr::scan_axis`] visits each lane.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScanDirection {
    Forward,
    Backward,
}

/// Output shape of a lane reduction: `axis` removed, or kept with extent 1.
/// The lanes are visited in C order over the other axes either way, so the
/// values need no reordering.
fn reduced_dim<X: ReduceAxis<R>, R: Rank>(shape: &[usize], axis: usize) -> Dim<X::Output> {
    let mut out = shape.to_vec();
    if X::KEEP {
        out[axis] = 1;
    } else {
        out.remove(axis);
    }
    Dim::new(&out).expect("the axis argument selects the output rank")
}

fn contiguous_lane_start(shape: &[usize], axis: usize, mut lane: usize) -> usize {
    let mut position = 0;
    let mut stride = 1;
    for current_axis in (0..shape.len()).rev() {
        if current_axis == axis {
            stride *= shape[current_axis];
            continue;
        }
        let extent = shape[current_axis];
        position += (lane % extent) * stride;
        lane /= extent;
        stride *= extent;
    }
    position
}

impl<T, R: Rank, B: Buffer<T>> Ndarr<T, R, B> {
    /// Materialize logical C-order elements into an owned array.
    pub fn to_owned_array(&self) -> Ndarr<T, R>
    where
        T: Clone,
    {
        Ndarr::contiguous(self.iter_elems().cloned().collect(), self.dim.clone())
    }

    /// Map elements in logical C-order. Borrows each element directly from
    /// the buffer, so non-`Copy` elements are never cloned.
    pub fn map<T2, F: Fn(&T) -> T2>(&self, f: F) -> Ndarr<T2, R> {
        let out: Vec<T2> = self.iter_elems().map(f).collect();
        Ndarr::contiguous(out, self.dim.clone())
    }

    /// Transform every lane parallel to `axis`; each transformed lane must
    /// retain the input lane length.
    pub fn map_lanes<T2, F>(&self, axis: usize, mut f: F) -> Result<Ndarr<T2, R>, DimError>
    where
        T2: Clone,
        F: FnMut(NdView<'_, T, U1>) -> Ndarr<T2, U1>,
    {
        let lanes = self.lanes(axis)?;
        let lane_len = self.shape()[axis];

        if self.is_empty() {
            for lane in lanes {
                if f(lane).shape() != [lane_len] {
                    return Err(DimError::new("map_lanes must preserve lane length"));
                }
            }
            return Ok(Ndarr::contiguous(Vec::new(), self.dim.clone()));
        }

        let lane_stride: usize = self.shape()[axis + 1..].iter().product();
        let mut output = None;
        for (lane_index, lane) in lanes.enumerate() {
            let transformed = f(lane);
            if transformed.shape() != [lane_len] {
                return Err(DimError::new("map_lanes must preserve lane length"));
            }
            let values = transformed.into_data();
            let out = output.get_or_insert_with(|| vec![values[0].clone(); self.len()]);
            let start = contiguous_lane_start(self.shape(), axis, lane_index);
            for (index, value) in values.into_iter().enumerate() {
                out[start + index * lane_stride] = value;
            }
        }
        Ok(Ndarr::contiguous(
            output.expect("a non-empty array has at least one lane"),
            self.dim.clone(),
        ))
    }

    /// Apply `f` to every cell formed by the trailing `k` axes, collecting one
    /// result per position of the leading frame axes in logical C-order.
    /// Cells are borrowed views, so overlapping [`Win`](crate::Win) cells are
    /// never copied; use `permute_axes` to bring other axes to the end. A
    /// typenum `k` keeps fixed ranks; a `usize` `k` makes both `Dyn`. Empty
    /// cells reach `f`; an empty frame never calls it.
    ///
    /// ```
    /// use rapl::*;
    /// // Max pooling that also reports the row-major winner in each window.
    /// let x = Ndarr::from([[1, 5, 2], [4, 3, 9], [0, 8, 6]]);
    /// let windows = x.slice(s![Win(2), Win(2)]).unwrap(); // [row, kh, col, kw]
    /// let windows = windows.view().permute_axes(&[0, 2, 1, 3]).unwrap(); // [row, col, kh, kw]
    /// let winners = windows
    ///     .map_cells(U2::new(), |window| {
    ///         let mut items = window.iter_elems().copied().enumerate();
    ///         items.reduce(|best, x| if x.1 > best.1 { x } else { best }).unwrap()
    ///     })
    ///     .unwrap();
    /// assert_eq!(winners.map(|&(_, value)| value), Ndarr::from([[5, 9], [8, 9]]));
    /// assert_eq!(winners.map(|&(at, _)| at), Ndarr::from([[1, 3], [3, 1]]));
    /// ```
    /// ```compile_fail,E0277
    /// use rapl::*;
    /// let row = Ndarr::from([1, 2, 3]);
    /// row.map_cells(U2::new(), |cell| cell.len());
    /// ```
    pub fn map_cells<K, T2, F>(&self, k: K, f: F) -> Result<Ndarr<T2, K::Frame>, DimError>
    where
        K: CellAxes<R>,
        F: FnMut(NdView<'_, T, K::Cell>) -> T2,
    {
        let split = self
            .rank()
            .checked_sub(k.count())
            .ok_or_else(|| DimError::new("Cell axes exceed the array rank"))?;
        let (frame, cell) = self.shape().split_at(split);
        let (frame_strides, cell_strides) = self.strides().split_at(split);
        let frame_dim = Dim::new(frame)?;
        let cells = self.cells(
            frame,
            frame_strides,
            Dim::new(cell)?,
            rank_strides::<K::Cell>(cell_strides),
        );
        Ok(Ndarr::contiguous(cells.map(f).collect(), frame_dim))
    }

    /// Fold borrowed elements in each lane parallel to `axis`. A `usize` axis
    /// is removed; [`Keep`](crate::Keep)`(axis)` stays with extent 1 so the
    /// result broadcasts back against `self`. Empty lanes yield a copy of `init`.
    pub fn fold_axis<X, A, F>(
        &self,
        axis: X,
        init: A,
        f: F,
    ) -> Result<Ndarr<A, X::Output>, DimError>
    where
        X: ReduceAxis<R>,
        A: Clone,
        F: Fn(A, &T) -> A,
    {
        let axis = axis.axis();
        let values = self
            .lanes(axis)?
            .map(|lane| lane.iter_elems().fold(init.clone(), &f))
            .collect();
        Ok(Ndarr::contiguous(
            values,
            reduced_dim::<X, R>(self.shape(), axis),
        ))
    }

    /// Seedless reduction along an axis. A
    /// `usize` axis is removed; [`Keep`](crate::Keep)`(axis)` stays with extent 1.
    /// An empty axis is an error.
    pub fn reduce<X, F: Fn(T, T) -> T>(
        &self,
        axis: X,
        f: F,
    ) -> Result<Ndarr<T, X::Output>, DimError>
    where
        X: ReduceAxis<R>,
        T: Clone,
    {
        let axis = axis.axis();
        if axis >= self.rank() {
            return Err(DimError::new("Axis greater than rank"));
        }
        if self.shape()[axis] == 0 {
            return Err(DimError::new("Cannot reduce an empty axis"));
        }
        let values = self
            .lanes(axis)?
            .map(|lane| {
                lane.iter_elems()
                    .cloned()
                    .reduce(&f)
                    .expect("the reduction axis is non-empty")
            })
            .collect();
        Ok(Ndarr::contiguous(
            values,
            reduced_dim::<X, R>(self.shape(), axis),
        ))
    }

    /// Scan independently along every lane parallel to `axis`; the combining
    /// function receives `(accumulator, element)`.
    pub fn scan_axis<F>(
        &self,
        axis: usize,
        direction: ScanDirection,
        f: F,
    ) -> Result<Ndarr<T, R>, DimError>
    where
        T: Clone,
        F: Fn(T, T) -> T,
    {
        self.map_lanes(axis, |lane| {
            let dim = lane.dim().clone();
            let mut values: Vec<T> = lane.iter_elems().cloned().collect();
            match direction {
                ScanDirection::Forward => {
                    for index in 1..values.len() {
                        values[index] = f(values[index - 1].clone(), values[index].clone());
                    }
                }
                ScanDirection::Backward => {
                    for index in (0..values.len().saturating_sub(1)).rev() {
                        values[index] = f(values[index + 1].clone(), values[index].clone());
                    }
                }
            }
            Ndarr::contiguous(values, dim)
        })
    }

    /// Roll elements along `axis` by `shift`, wrapping around. Wrapping is
    /// not representable by one offset/stride tuple, so the result materializes.
    pub fn roll(&self, shift: isize, axis: usize) -> Ndarr<T, R>
    where
        T: Clone,
    {
        assert!(axis < self.rank(), "axis out of bounds");
        // Nothing to move when any extent is zero.
        if self.is_empty() {
            return self.to_owned_array();
        }
        let len = self.shape()[axis] as isize;
        let split = (len - shift.rem_euclid(len)) % len;
        let mut head: Vec<SliceSpec> = vec![(..).into(); self.rank()];
        let mut tail = head.clone();
        head[axis] = (split..).into();
        tail[axis] = (..split).into();
        let rolled = Ndarr::concatenate(
            axis,
            &[
                self.slice(head.as_slice()).expect("axis is in bounds"),
                self.slice(tail.as_slice()).expect("axis is in bounds"),
            ],
        )
        .expect("two slices of one array share every non-axis extent");
        Ndarr::contiguous(rolled.into_data(), self.dim.clone())
    }
}

impl<T: Clone, R: Rank> Ndarr<T, R> {
    /// Concatenate arrays along an existing axis in logical order.
    pub fn concatenate<B: Buffer<T>>(
        axis: usize,
        arrays: &[Ndarr<T, R, B>],
    ) -> Result<Self, DimError> {
        let first = arrays
            .first()
            .ok_or_else(|| DimError::new("concatenate requires at least one array"))?;
        if axis >= first.rank() {
            return Err(DimError::new("Axis greater than rank"));
        }
        if arrays.iter().any(|array| {
            array.rank() != first.rank()
                || array
                    .shape()
                    .iter()
                    .zip(first.shape())
                    .enumerate()
                    .any(|(current_axis, (left, right))| current_axis != axis && left != right)
        }) {
            return Err(DimError::new(
                "concatenate requires matching non-axis extents",
            ));
        }

        let axis_len = arrays.iter().try_fold(0usize, |total, array| {
            total
                .checked_add(array.shape()[axis])
                .ok_or_else(|| DimError::new("concatenated axis length overflow"))
        })?;
        let mut shape = first.shape().to_vec();
        shape[axis] = axis_len;
        let dim = Dim::new(&shape)?;
        let outer: usize = first.shape()[..axis].iter().product();
        let inner: usize = first.shape()[axis + 1..].iter().product();
        let mut iterators: Vec<_> = arrays.iter().map(Ndarr::iter_elems).collect();
        let mut data = Vec::with_capacity(dim.get_number_elements());
        for _ in 0..outer {
            for (array, elements) in arrays.iter().zip(&mut iterators) {
                let block = array.shape()[axis]
                    .checked_mul(inner)
                    .ok_or_else(|| DimError::new("concatenated block length overflow"))?;
                data.extend(elements.by_ref().take(block).cloned());
            }
        }
        Ok(Ndarr::contiguous(data, dim))
    }
}

impl<T, R: Rank, B: BufferMut<T>> Ndarr<T, R, B> {
    pub fn map_in_place<F: Fn(&T) -> T>(&mut self, f: F) {
        let positions = PosIter::new(self.shape(), self.strides(), self.offset, self.len());
        let buffer = self.buffer.as_mut_slice();
        for p in positions {
            buffer[p] = f(&buffer[p]);
        }
    }

    /// Replace each element with `f(element, matching element of other)`.
    /// `other` broadcasts to `self`'s shape and may hold another element type,
    /// so a mask or a bias updates in place.
    ///
    /// ```
    /// use rapl::*;
    /// let mut scores = Ndarr::from([[[1.0, 2.0], [3.0, 4.0]]]); // [batch, query, key]
    /// let future = Ndarr::from_fn([2, 2], |ix| ix[1] > ix[0]); // broadcast over the batch
    /// scores.zip_with_in_place(&future, |&s, &hide| if hide { f64::NEG_INFINITY } else { s })?;
    /// assert_eq!(scores, Ndarr::from([[[1.0, f64::NEG_INFINITY], [3.0, 4.0]]]));
    /// # Ok::<(), DimError>(())
    /// ```
    pub fn zip_with_in_place<T2, R2, B2, F>(
        &mut self,
        other: &Ndarr<T2, R2, B2>,
        f: F,
    ) -> Result<(), DimError>
    where
        R2: Rank,
        B2: Buffer<T2>,
        F: Fn(&T, &T2) -> T,
    {
        let other = other.broadcast_view_to(&self.dim)?;
        let positions = PosIter::new(self.shape(), self.strides(), self.offset, self.len());
        let buffer = self.buffer.as_mut_slice();
        for (p, r) in positions.zip(other.iter_elems()) {
            buffer[p] = f(&buffer[p], r);
        }
        Ok(())
    }
}
