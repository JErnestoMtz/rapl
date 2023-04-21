use super::*;
use num_traits::Signed;
use std::ops::Add;

impl<T, const R: usize> Ndarr<T, R>
where
    T: Clone + Debug + Default + Signed + PartialOrd,
{
    pub fn abs(&self) -> Self {
        let out = self.map(|x| x.abs());
        out
    }
    pub fn is_positive(&self) -> Ndarr<bool, R> {
        let out = self.map_types(|x| x.is_positive());
        out
    }

    pub fn is_negative(&self) -> Ndarr<bool, R> {
        let out = self.map_types(|x| x.is_negative());
        out
    }
}

impl<T, const R: usize> Ndarr<T, R>
where
    T: Clone + Debug + Default,
{
    pub fn sum(&self) -> T 
        where T :Add<Output = T>,
    {
        let data = &self.data;
        let mut sum = data[0].clone();
        for i in 1..data.len() {
            sum = sum + data[i].clone();
        }
        return sum;
    }

    pub fn max(&self) -> Option<&T>
        where T : Ord,
    {
        self.data.iter().max()
    }
}


#[cfg(test)]
mod test_extras{
    use super::Ndarr;

    #[test]
    fn max(){
        let arr = Ndarr::from([-2,0,4,8]);
        assert_eq!(arr.max().unwrap(), &8)
    }

    #[test]
    fn sum(){
        let arr = Ndarr::from([-2,0,4,8]);
        assert_eq!(arr.sum(), 10)
    }
}


