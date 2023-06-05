use super::*;

impl<T1: Clone, R: Unsigned> Ndarr<T1, R> {
    pub fn map_in_place<F1: Fn(&T1) -> T1>(&mut self, f: F1) {
        for val in self.data.iter_mut() {
            *val = f(val)
        }
    }
    pub fn map<T2: Clone + Debug, F2: Fn(&T1) -> T2>(&self, f: F2) -> Ndarr<T2, R> {
        let out: Vec<T2> = self.data.iter().map(f).collect();
        Ndarr {
            data: out,
            dim: self.dim.clone(),
        }
    }
    // Bimap: is the same as Zip then map, is just a convenient way for doing diadic operations between Ndarrs
    pub fn bimap<F: Fn(T1, T1) -> T1>(&self, other: &Self, f: F) -> Self 
    where T1: Default,
    {
        let mut out = Vec::with_capacity(self.data.len());
        for i in 0..self.data.len(){
            out.push(f(self.data[i].clone(), other.data[i].clone()))
        }
        Ndarr {
            data: out,
            dim: self.dim.clone(),
        }
    }

    pub fn bimap_in_place<F: Fn(T1, T1) -> T1>(&mut self, other: &Self, f: F) {
        for i in 0..self.data.len() {
            self.data[i] = f(self.data[i].clone(), other.data[i].clone())
        }
    }

    pub fn scanr<F: Fn(T1, T1) -> T1>(&self, axis: usize, f: F) -> Self
    where
        T1: Default,
        R: Sub<B1>,
        <R as Sub<B1>>::Output: Unsigned,
        <R as Sub<B1>>::Output: Add<B1>,
        <<R as Sub<B1>>::Output as Add<B1>>::Output: Unsigned,
    {
        let mut slices = self.slice_at(axis);
        for i in 0..slices.len() - 1 {
            slices[i + 1] = slices[i + 1].bimap(&slices[i], &f)
        }
        let out = de_slice(&slices, axis);
        Ndarr {
            data: out.data,
            dim: self.dim.clone(),
        }
    }

    pub fn scanl<F: Fn(T1, T1) -> T1>(&self, axis: usize, f: F) -> Self
    where
        T1: Default,
        R: Sub<B1>,
        <R as Sub<B1>>::Output: Unsigned,
        <R as Sub<B1>>::Output: Add<B1>,
        <<R as Sub<B1>>::Output as Add<B1>>::Output: Unsigned,
    {
        let mut slices = self.slice_at(axis);
        let l = slices.len();
        for i in 0..slices.len() - 1 {
            slices[l - 2 - i] = slices[l - 2 - i].bimap(&slices[l - 1 - i], &f)
        }
        let out = de_slice(&slices, axis);
        Ndarr {
            data: out.data,
            dim: self.dim.clone(),
        }
    }
}
