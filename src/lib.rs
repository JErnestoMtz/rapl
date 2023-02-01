mod helpers;
use std::{fmt::Debug};


#[derive(Debug, Copy, Clone)]
pub struct Ndarr<T: Copy + Clone + Default, const N: usize, const R: usize>{
    pub data: [T; N],
    pub rank: [usize; R]
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
       Ndarr { data: out , rank: self.rank }
    }
}

trait Transpose {
    fn T(self) -> Self;
}

// Generic transpose for array of rank R
    // The helper functions use in here can be derive with some maths, but maybe there is a beeter way to do it.
impl <T: Default + Copy + Clone, const N: usize, const R: usize>Transpose for Ndarr<T, N, R>
    where [T; N]: Default,
    [usize; R]: Default
    
{
    fn T(self) -> Self {
        let shape = self.rank.clone();
        let mut out_dim: [usize; R] = self.rank.clone();
        out_dim.reverse();
        let mut out_arr: [T; N] = Default::default();
        for i in 0..N{
            let mut new_indexes = helpers::get_indexes(&i, &shape);
            new_indexes.reverse();
            let new_pos = helpers::get_flat_pos(&new_indexes, &out_dim).unwrap();
            out_arr[new_pos] = self.data[i].clone();
        }
        Ndarr{data: out_arr, rank: out_dim}
    }
    
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
    }
}
