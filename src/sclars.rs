use super::*;

pub fn extend_scalar<P,T, const N: usize, const R: usize>(scalar: P, ndarr: &Ndarr<T, N, R>)-> Ndarr<T,N,R>
    where T: Debug + Copy + Clone + Default,
    P: Into<T> + Clone,
    [T; N]: Default
{
   let mut out_data: [T; N] = Default::default();
   for i in 0..N{
    out_data[i] = scalar.clone().into();
   };
   Ndarr { data: out_data, shape: ndarr.shape.clone() }

}