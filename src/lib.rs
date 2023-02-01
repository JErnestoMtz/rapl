mod helpers;
use std::{fmt::Debug, collections::hash_map::RandomState};

// main struct of N Dimentional generic array.
    //the shape is denoted by the `shape` array where the length is the Rank of the Ndarray
    //the actual values are stored in a flattend state in a rank 1 array

#[derive(Debug, Copy, Clone)]
pub struct Ndarr<T: Copy + Clone + Default, const N: usize, const R: usize>{
    pub data: [T; N],
    pub shape: [usize; R]
}

impl  <T: Copy + Clone + Debug + Default, const N: usize, const R: usize>Ndarr<T,N,R> {
    //TODO: implement errors 
    pub fn new(data: [T; N], shape: [usize; R]) -> Result<Self,String>{
        let n = helpers::muliply_list(&shape, 1);
        if data.len() ==  n{
            Ok(Ndarr { data: data, shape: shape })
        }else {
            Err(format!("The number of elements of an Ndarray of shape {:?} is {}, and {} were provided.", shape, n, N))
        }
    }
    pub fn rank(self)-> usize{
        R
    }
    pub fn shape(self)->[usize; R]{
        self.shape
    }
    
}
trait Bimap<F>{
    fn bimap(self, other: Self, f: F)-> Self;
}

// Here we need to think about if valueble maybe checking for the same shape and return an option instead
impl <F, T: Copy + Debug + Clone + Default, const N: usize, const R: usize>Bimap<F> for Ndarr<T, N, R> 
    where F: Fn(&T,&T)->T,
    [T; N]: Default,
{
    fn bimap(self, other: Self, f: F)-> Self {
       let mut out: [T; N] = Default::default();
        for i in 0..N{
            out[i] = f(&self.data[i], &other.data[i])
        }
       Ndarr { data: out , shape: self.shape }
    }

}

trait Transpose {
    fn t(self) -> Self;
}

// Generic transpose for array of rank R
    // The helper functions use in here can be derive with some maths, but maybe there is a beeter way to do it.
impl <T: Default + Copy + Clone, const N: usize, const R: usize>Transpose for Ndarr<T, N, R>
    where [T; N]: Default,
    [usize; R]: Default
    
{
    fn t(self) -> Self {
        let shape = self.shape.clone();
        let mut out_dim: [usize; R] = self.shape.clone();
        out_dim.reverse();
        let mut out_arr: [T; N] = Default::default();
        for i in 0..N{
            let mut new_indexes = helpers::get_indexes(&i, &shape);
            new_indexes.reverse();
            let new_pos = helpers::get_flat_pos(&new_indexes, &out_dim).unwrap();
            out_arr[new_pos] = self.data[i].clone();
        }
        Ndarr{data: out_arr, shape: out_dim}
    }
    
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constoctor_test() {
        let arr = Ndarr::new([0, 1, 2, 3], [2,2]).expect("Error initializing");
        assert_eq!(arr.shape(), [2,2]);
        assert_eq!(arr.rank(), 2)
    }

    #[test]
    fn bimap_test() {
        let arr1 = Ndarr::new([0, 1, 2, 3], [2,2]).expect("Error initializing");
        let arr2 = Ndarr::new([4, 5, 6, 7], [2,2]).expect("Error initializing");
        assert_eq!(arr1.bimap(arr2, |x, y| x + y).data, [4, 6, 8, 10])
    }


}
