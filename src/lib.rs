mod helpers;
mod ops;
use std::{
    fmt::Debug,
    fmt::{write, Display},
};

// main struct of N Dimensional generic array.
//the shape is denoted by the `shape` array where the length is the Rank of the Ndarray
//the actual values are stored in a flattend state in a rank 1 array

#[derive(Debug, Copy, Clone)]
pub struct Ndarr<T: Copy + Clone + Default, const N: usize, const R: usize> {
    pub data: [T; N],
    pub shape: [usize; R],
}

impl<T: Copy + Clone + Debug + Default, const N: usize, const R: usize> Ndarr<T, N, R> {
    //TODO: implement errors
    pub fn new(data: [T; N], shape: [usize; R]) -> Result<Self, String> {
        let n = helpers::multiply_list(&shape, 1);
        if data.len() == n {
            Ok(Ndarr {
                data: data,
                shape: shape,
            })
        } else {
            Err(format!(
                "The number of elements of an Ndarray of shape {:?} is {}, and {} were provided.",
                shape, n, N
            ))
        }
    }
    pub fn rank(self) -> usize {
        R
    }
    pub fn shape(self) -> [usize; R] {
        self.shape
    }
}

impl<T: Copy + Clone + Debug + Default + Display, const N: usize, const R: usize> Display
    for Ndarr<T, N, R>
{
    // Kind of nasty function, it can be imprube a lot, but I think there is no scape from recursion.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let strs = self.data.map(|x| x.to_string());
        let binding = strs.clone().map(|s| s.len());
        let max_size = binding.iter().max().unwrap();
        let mut fmt_str: [String; N] = strs.map(|s| helpers::format_vla(s, max_size));
        let mut splits = self.shape.clone();
        splits.reverse();

        fn slip_format<'a>(strings: &'a mut [String], splits: &'a [usize]) -> () {
            if splits.len() == 0 {
                return;
            }
            let l = helpers::multiply_list(splits, 1);
            let n_splits = strings.len() / l;
            for i in 0..n_splits {
                let new_s: &mut [String] = &mut strings[i * l..(i + 1) * l];
                new_s[0].insert_str(0, "[");
                new_s[l - 1].push_str("]");
                slip_format(new_s, &splits[1..]);
            }
            return;
        }
        // TODO: add new lines in the correct places to display it more numpy like
        slip_format(&mut fmt_str[0..], &mut splits[..]);

        let out = fmt_str.clone().join(" ");
        write!(f, "Ndarr({})", out)
    }
}
trait Bimap<F> {
    fn bimap(self, other: Self, f: F) -> Self;
}

// Here we need to think about if valueble maybe checking for the same shape and return an option instead
impl<F, T: Copy + Debug + Clone + Default, const N: usize, const R: usize> Bimap<F>
    for Ndarr<T, N, R>
where
    F: Fn(&T, &T) -> T,
    [T; N]: Default,
{
    fn bimap(self, other: Self, f: F) -> Self {
        let mut out: [T; N] = Default::default();
        for i in 0..N {
            out[i] = f(&self.data[i], &other.data[i])
        }
        Ndarr {
            data: out,
            shape: self.shape,
        }
    }
}



trait Map<F> {
    fn map(self, f: F) -> Self;
    
    fn mapinplace(&mut self, f: F);
}

// Here we need to think about if worth it maybe checking for the same shape and return an option instead or just panic()
impl<F, T: Copy + Debug + Clone + Default, const N: usize, const R: usize> Map<F>
    for Ndarr<T, N, R>
where
    F: Fn(&T) -> T,
    [T; N]: Default,
{
    fn map(self, f: F) -> Self{
        let mut out: [T; N] = Default::default();
        for i in 0..N {
            out[i] = f(&self.data[i])
        }
        Ndarr {
            data: out,
            shape: self.shape,
        }
    }
    fn mapinplace(&mut self, f: F){
       for i in 0..N{
            self.data[i] = f(&self.data[i])
       } 
    }
}


trait Transpose {
    fn t(self) -> Self;
}

// Generic transpose for array of rank R
    // the basic idea of a generic transpose of an N-dimentional array is to flip de shape of it like in a mirror.
    // The helper functions use in here can be derive with some maths, but maybe there is a beeter way to do it.
impl<T: Default + Copy + Clone, const N: usize, const R: usize> Transpose for Ndarr<T, N, R>
where
    [T; N]: Default,
    [usize; R]: Default,
{
    fn t(self) -> Self {
        let shape = self.shape.clone();
        let mut out_dim: [usize; R] = self.shape.clone();
        out_dim.reverse();
        let mut out_arr: [T; N] = Default::default();
        for i in 0..N {
            let mut new_indexes = helpers::get_indexes(&i, &shape);
            new_indexes.reverse();
            let new_pos = helpers::get_flat_pos(&new_indexes, &out_dim).unwrap();
            out_arr[new_pos] = self.data[i].clone();
        }
        Ndarr {
            data: out_arr,
            shape: out_dim,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constructor_test() {
        let arr = Ndarr::new([0, 1, 2, 3], [2, 2]).expect("Error initializing");
        assert_eq!(arr.shape(), [2, 2]);
        assert_eq!(arr.rank(), 2)
    }

    #[test]
    fn bimap_test() {
        let arr1 = Ndarr::new([0, 1, 2, 3], [2, 2]).expect("Error initializing");
        let arr2 = Ndarr::new([4, 5, 6, 7], [2, 2]).expect("Error initializing");
        assert_eq!(arr1.bimap(arr2, |x, y| x + y).data, [4, 6, 8, 10])
    }

    #[test]
    fn transpose() {
        let arr = Ndarr::new([0, 1, 2, 3, 4, 5, 6, 7], [2, 2, 2]).expect("Error initializing");
        println!("{}", arr);
        // same as arr.T.flatten() in numpy
        assert_eq!(arr.t().data, [0, 4, 2, 6, 1, 5, 3, 7])
    }

    #[test]
    fn element_wise_ops() {
        let arr1 = Ndarr::new([1, 1, 1, 1], [2, 2]).expect("Error initializing");
        let arr2 = Ndarr::new([1, 1, 1, 1], [2, 2]).expect("Error initializing");

        let arr3 = Ndarr::new([2, 2, 2, 2], [2, 2]).expect("Error initializing");
        assert_eq!((arr1 + arr2).data, arr3.data);
        assert_eq!((arr1 - arr2).data, [0,0,0,0]);
        assert_eq!((arr3 * arr3).data, [4,4,4,4]);
        assert_eq!((arr3 / arr3).data, [2,2,2,2]);
    }
}
