use num_traits::Float;

use super::*;

impl <T, const R: usize> Ndarr<T,R> 
where T: Clone + Copy + Debug + Default + Float
{
    pub fn sin(&self)->Self{
        let out = self.map(|x| x.sin());
        out
    }

    pub fn cos(&self)->Self{
        let out = self.map(|x| x.cos());
        out
    }

    pub fn tan(&self)->Self{
        let out = self.map(|x| x.tan());
        out
    }

    pub fn sinh(&self)->Self{
        let out = self.map(|x| x.sinh());
        out
    }
    
    pub fn cosh(&self)->Self{
        let out = self.map(|x| x.cosh());
        out
    }

    pub fn tanh(&self)->Self{
        let out = self.map(|x| x.tanh());
        out
    }

    pub fn log(&self, base: T)->Self{
        let out = self.map(|x| x.log(base));
        out
    }
    
    pub fn ln(&self)->Self{
        let out = self.map(|x| x.ln());
        out
    }

    pub fn log2(&self)->Self{
        let out = self.map(|x| x.log2());
        out
    }
    pub fn is_infinite(&self) -> Ndarr<bool,R> {
       let out = self.map_types(|x| x.is_infinite() );
       out
    }
    pub fn is_finite(&self) -> Ndarr<bool,R> {
       let out = self.map_types(|x| x.is_finite() );
       out
    }
    pub fn is_normal(&self) -> Ndarr<bool, R> {
       let out = self.map_types(|x| x.is_normal() );
       out

    }
    pub fn is_nan(&self) -> Ndarr<bool,R>{
       let out = self.map_types(|x| x.is_nan() );
       out
    }
}
