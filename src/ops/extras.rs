
use super::*;
use std::ops::Add;
use num_traits::Signed;


impl <T, const R: usize> Ndarr<T,R> 
where T: Clone + Debug + Default + Signed + PartialOrd
{
    pub fn abs(&self)->Self{
       let out = self.map(|x| x.abs());
       out
    }
    pub fn is_positive(&self)->Ndarr<bool, R>{
       let out = self.map_types(|x| x.is_positive());
       out
    }

    pub fn is_negative(&self)->Ndarr<bool, R>{
       let out = self.map_types(|x| x.is_negative());
       out
    }
}

impl <T, const R: usize> Ndarr<T,R> 
where T: Clone + Debug + Default + Add<Output = T> + PartialOrd
{
    pub fn sum(&self)->T{
        let data = self.data.clone();
        let mut sum = data[0].clone();
        for t in data{
            sum = sum + t;
        }
        return sum;
    }
}