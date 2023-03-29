use super::*;

trait GeneralBimap<F, T2, T3> {
    type Other;
    type Output;
    fn gen_bimap(self, other: Self::Other, f: F) -> Self::Output;
}

impl<F, T1, T2, T3, const R: usize> GeneralBimap<F, T2, T3> for Ndarr<T1, R>
where
    T1: Debug + Clone + Default,
    T2: Debug + Clone + Default,
    T3: Debug + Clone + Default,
    F: Fn(T1, T2) -> T3,
{
    type Other = Ndarr<T2, R>;
    type Output = Ndarr<T3, R>;

    fn gen_bimap(self, other: Self::Other, f: F) -> Self::Output {
        let mut out = vec![T3::default(); self.data.len()];
        for (i, val) in out.iter_mut().enumerate() {
            *val = f(self.data[i].clone(), other.data[i].clone())
        }
        Ndarr {
            data: out,
            shape: self.shape,
        }
    }
}




impl<T1: Debug + Clone + Default, const R: usize> Ndarr<T1, R>{
    pub fn map<F1: Fn(&T1)->T1>(&self, f: F1) -> Self{
        let mut out = vec![T1::default(); self.data.len()];
        for (i, val) in out.iter_mut().enumerate() {
            *val = f(&self.data[i])
        }
        Ndarr {
            data: out,
            shape: self.shape,
        }
    }
    pub fn map_in_place<F1: Fn(&T1)->T1>(&mut self, f: F1) {
        for val in self.data.iter_mut() {
            *val = f(val)
        }
    }
    pub fn map_types<T2: Clone + Debug + Default, F2: Fn(&T1)->T2>(&self, f: F2)->Ndarr<T2,R>{
        let mut out = vec![T2::default(); self.data.len()];
        for (i, val) in out.iter_mut().enumerate() {
            *val = f(&self.data[i])
        }
        Ndarr {
            data: out,
            shape: self.shape,
        }
    }
    // Bimap: is the same as Zip then map, is just a convenient way for doing diadic operations between Ndarrs
    pub fn bimap<F: Fn(T1,T1)->T1>(&self, other: &Self, f: F) -> Self {
        let mut out = vec![T1::default(); self.data.len()];
        for (i, val) in out.iter_mut().enumerate() {
            *val = f(self.data[i].clone(), other.data[i].clone())
        }
        Ndarr {
            data: out,
            shape: self.shape,
        }
    }

    pub fn bimap_in_place<F: Fn(T1,T1)->T1>(&mut self, other: &Self, f: F) {
        for i in 0..self.data.len() {
            self.data[i] = f(self.data[i].clone(), other.data[i].clone())
        }
    }


}