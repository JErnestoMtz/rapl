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

    pub fn scanr<F: Fn(T1,T1)->T1>(&self,axis: usize, f: F) -> Self
        where [usize; R -1]: Sized,
        [usize; R - 1 + 1]: Sized, // lol const generics still too dum, we need this or it else breaks
        [usize; R + 1 - 1]: Sized, 
    {
        let mut slices = self.slice_at(axis);
        for i in 0..slices.len()-1{
            slices[i + 1] = slices[i+1].bimap(&slices[i], &f)
        }
        let out = de_slice(&slices, axis);
        Ndarr{data: out.data, shape: self.shape}
    }

    pub fn scanl<F: Fn(T1,T1)->T1>(&self,axis: usize, f: F) -> Self
        where [usize; R -1]: Sized,
        [usize; R - 1 + 1]: Sized, // lol const generics still too dum, we need this or it else breaks
        [usize; R + 1 - 1]: Sized,
    {
        let mut slices = self.slice_at(axis);
        let l = slices.len();
        for i in 0..slices.len()-1{
            slices[l - 2 -i] = slices[l- 2 -i].bimap(&slices[l - 1 - i], &f)
        }
        let out = de_slice(&slices, axis);
        Ndarr{data: out.data, shape: self.shape}
    }

}