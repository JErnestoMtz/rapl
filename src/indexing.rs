use super::*;
use std::ops::{Index, IndexMut};

/// `a[[i, j]]` spells element access for every rank. A fixed rank rejects the
/// wrong coordinate count when the call is compiled; [`Dyn`] checks it in
/// [`Ndarr::flat_index`], the one offset calculation behind every form.
const fn coordinate_count<R: Rank, const N: usize>() {
    if let Some(rank) = R::FIXED_RANK {
        assert!(rank == N, "coordinate count does not match the fixed rank");
    }
}

impl<T, R: Rank, B: Buffer<T>, const N: usize> Index<[usize; N]> for Ndarr<T, R, B> {
    type Output = T;
    fn index(&self, index: [usize; N]) -> &T {
        const { coordinate_count::<R, N>() };
        &self[index.as_slice()]
    }
}

impl<T, R: Rank, B: BufferMut<T>, const N: usize> IndexMut<[usize; N]> for Ndarr<T, R, B> {
    fn index_mut(&mut self, index: [usize; N]) -> &mut T {
        const { coordinate_count::<R, N>() };
        &mut self[index.as_slice()]
    }
}

/// Coordinates known only at runtime, such as a `Vec` built for a [`Dyn`] array.
impl<T, R: Rank, B: Buffer<T>> Index<&[usize]> for Ndarr<T, R, B> {
    type Output = T;
    fn index(&self, index: &[usize]) -> &T {
        &self.buffer.as_slice()[self.flat_index(index).unwrap()]
    }
}

impl<T, R: Rank, B: BufferMut<T>> IndexMut<&[usize]> for Ndarr<T, R, B> {
    fn index_mut(&mut self, index: &[usize]) -> &mut T {
        let flat_pos = self.flat_index(index).unwrap();
        &mut self.buffer.as_mut_slice()[flat_pos]
    }
}

#[cfg(test)]
mod indexing_tes {
    use super::*;
    use crate::{s, U1};

    #[test]
    fn indexing() {
        let mut arr = Ndarr::from([[1, 2], [3, 4]]);
        assert_eq!(arr[[0, 0]], 1);
        arr[[0, 1]] = 8;
        arr[[1, 1]] = 10;
        assert_eq!(&arr, &Ndarr::from([[1, 8], [3, 10]]));
    }

    #[test]
    fn fixed_and_dynamic_ranks_share_one_spelling() {
        let mut fixed = Ndarr::from([[1, 2, 3], [4, 5, 6]]);
        let mut dynamic = fixed.clone().into_dyn();
        assert_eq!(dynamic[[1, 2]], fixed[[1, 2]]);
        dynamic[[0, 1]] = 20;
        fixed[[0, 1]] = 20;
        assert_eq!(dynamic, fixed);
        let runtime = vec![1, 0];
        assert_eq!(dynamic[runtime.as_slice()], 4);
        dynamic[runtime.as_slice()] = 40;
        assert_eq!(dynamic[[1, 0]], 40);
    }

    #[test]
    fn views_index_through_their_strides() {
        let reverse = crate::Slice::stepped(.., -1);
        let mut arr = Ndarr::from([[1, 2, 3], [4, 5, 6]]);
        assert_eq!(arr.slice(s![.., reverse]).unwrap()[[0, 0]], 3);
        let dynamic = arr.clone().into_dyn();
        let specs = [SliceSpec::from(..), SliceSpec::from(reverse)];
        assert_eq!(dynamic.slice(&specs[..]).unwrap()[[1, 2]], 4);
        let mut column: NdViewMut<'_, i32, U1> = arr.slice_mut(s![.., 1]).unwrap();
        column[[1]] = 50;
        assert_eq!(arr[[1, 1]], 50);
    }

    #[test]
    fn dynamic_rank_checks_count_and_bounds_at_runtime() {
        let dynamic = Ndarr::from([[1, 2], [3, 4]]).into_dyn();
        let wrong_count = std::panic::catch_unwind(|| dynamic[[0, 0, 0]]);
        assert!(wrong_count.is_err());
        let out_of_bounds = std::panic::catch_unwind(|| dynamic[[2, 0]]);
        assert!(out_of_bounds.is_err());
        let fixed = Ndarr::from([[1, 2], [3, 4]]);
        assert!(std::panic::catch_unwind(|| fixed[[0, 2]]).is_err());
        assert!(std::panic::catch_unwind(|| fixed[[0, 0, 0].as_slice()]).is_err());
    }
}
