use num_traits::{One, Zero};

use super::*;
use crate::array::PosIter;

impl<T, R: Rank> Ndarr<T, R> {
    /// Build an array by calling `f` once per element in logical C-order,
    /// passing that element's coordinates. Ignore them with `|_|`.
    ///
    /// ```
    /// use rapl::*;
    /// let causal = Ndarr::from_fn([3, 3], |ix| ix[1] > ix[0]);
    /// assert_eq!(causal[[0, 2]], true);
    /// assert_eq!(causal[[2, 0]], false);
    /// let mut draws = 0..;
    /// let counted = Ndarr::<i32, U1>::from_fn([3], |_| draws.next().unwrap());
    /// assert_eq!(counted, Ndarr::from([0, 1, 2]));
    /// ```
    pub fn from_fn<D: Into<Dim<R>>, F>(shape: D, mut f: F) -> Self
    where
        F: FnMut(&[usize]) -> T,
    {
        let shape = shape.into();
        let len = shape.get_number_elements();
        // The shared odometer supplies coordinates; no buffer is walked yet.
        let mut walk = PosIter::new(shape.as_slice(), &vec![0; shape.len()], 0, len);
        let data = (0..len)
            .map(|_| {
                let value = f(walk.coordinates());
                walk.next();
                value
            })
            .collect();
        Ndarr::contiguous(data, shape)
    }
}

impl<T: Clone, R: Rank> Ndarr<T, R> {
    pub fn zeros<D: Into<Dim<R>>>(shape: D) -> Self
    where
        T: Zero,
    {
        Self::from_fn(shape, |_| T::zero())
    }

    pub fn ones<D: Into<Dim<R>>>(shape: D) -> Self
    where
        T: One,
    {
        Self::from_fn(shape, |_| T::one())
    }

    pub fn fill<D: Into<Dim<R>>>(with: T, shape: D) -> Self {
        Self::from_fn(shape, |_| with.clone())
    }
}

#[cfg(test)]
mod from_fn_test {
    use super::*;

    #[test]
    fn fills_in_c_order_with_exact_call_count() {
        let mut calls = 0;
        let counted = Ndarr::from_fn([2, 2], |_| {
            calls += 1;
            calls
        });
        assert_eq!(counted, Ndarr::from([[1, 2], [3, 4]]));
        assert_eq!(calls, 4);
    }

    #[test]
    fn passes_each_element_its_coordinates() {
        let shape = [2usize, 3, 4];
        let encode = |ix: &[usize]| ix.iter().fold(0, |acc, &i| acc * 10 + i);
        let fixed = Ndarr::<usize, U3>::from_fn(shape, encode);
        let dynamic = Ndarr::<usize, Dyn>::from_fn(&shape[..], encode);
        let mut visited = Vec::new();
        let recorded = Ndarr::<usize, U3>::from_fn(shape, |ix| {
            visited.push(ix.to_vec());
            0
        });
        assert_eq!(recorded.len(), 24);
        assert_eq!(visited.first(), Some(&vec![0, 0, 0]));
        assert_eq!(visited[1], vec![0, 0, 1]);
        assert_eq!(visited.last(), Some(&vec![1, 2, 3]));
        for coordinates in &visited {
            assert_eq!(fixed[coordinates.as_slice()], encode(coordinates));
            assert_eq!(dynamic[coordinates.as_slice()], encode(coordinates));
        }
        let identity = Ndarr::from_fn([3, 3], |ix| i32::from(ix[0] == ix[1]));
        assert_eq!(identity, Ndarr::from([[1, 0, 0], [0, 1, 0], [0, 0, 1]]));
    }

    #[test]
    fn handles_empty_shapes_and_rank_zero() {
        let mut calls = 0;
        let empty: Ndarr<i32, U2> = Ndarr::from_fn([2, 0], |_| {
            calls += 1;
            1
        });
        assert!(empty.is_empty());
        assert_eq!(calls, 0);

        let scalar = Ndarr::<i32, U0>::from_fn([], |ix| {
            assert!(ix.is_empty());
            7
        });
        assert_eq!(scalar.scalar(), 7);
    }

    #[test]
    fn from_fn_does_not_require_clone() {
        struct NotClone(i32);

        let mut value = 0;
        let array: Ndarr<NotClone, U1> = Ndarr::from_fn([2], |_| {
            value += 1;
            NotClone(value)
        });
        assert_eq!(
            array
                .into_data()
                .into_iter()
                .map(|item| item.0)
                .collect::<Vec<_>>(),
            vec![1, 2]
        );
    }

    #[test]
    fn zeros_ones_and_fill_are_generator_compositions() {
        assert_eq!(
            Ndarr::<i32, U2>::zeros([2, 2]),
            Ndarr::from_fn([2, 2], |_| 0)
        );
        assert_eq!(Ndarr::<i32, U1>::ones([3]), Ndarr::from_fn([3], |_| 1));
        assert_eq!(Ndarr::fill(5, [4]), Ndarr::from_fn([4], |_| 5));
    }
}
