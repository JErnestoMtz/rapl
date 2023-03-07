use super::*;
use crate::helpers::*;

#[derive(Debug)]
pub struct DimError {
    details: String,
}

impl DimError {
    pub fn new(msg: &str) -> DimError {
        DimError {
            details: msg.to_string(),
        }
    }
}

pub trait Slice {
    type Output;
    fn slice_at(self, axis: usize) -> Self::Output;
}

impl<T, const R: usize> Slice for Ndarr<T, R>
where
    T: Copy + Clone + Default + Debug,
    [usize;  R - 1 ]: Sized,
{
    type Output = Vec<Ndarr<T, { R - 1 }>>;
    fn slice_at(self, axis: usize) -> Self::Output {
        let n = multiply_list(&self.shape, 1); // number of elements in original array
        let new_shape: [usize;  R - 1 ] = remove_element(self.shape, axis);
        let n_new_arrs = self.shape[axis]; // number of new arrays

        let iota = 0..n;

        let indexes: Vec<[usize; R]> = iota.map(|i| get_indexes(&i, &self.shape)).collect(); //indexes of each element

        let mut out: Vec<Ndarr<T, { R - 1 }>> = Vec::new(); // to sore

        for i in 0..n_new_arrs {
            let mut this_data: Vec<T> = Vec::new();
            for j in 0..n {
                if indexes[j][axis] == i {
                    let ind = get_flat_pos(&indexes[j], &self.shape).unwrap();
                    this_data.push(self.data[ind])
                }
            }
            out.push(Ndarr::new(&this_data, new_shape).expect("Error initializing"))
        }
        out
    }
}
pub trait Reduce<F> {
    type Output;
    fn reduce(self, axis: usize, f: F) -> Self::Output;
}

impl<T, F, const R: usize> Reduce<F> for Ndarr<T, R>
where
    F: Fn(&T, &T) -> T + Clone,
    T: Copy + Clone + Default + Debug,
    [usize;  R - 1 ]: Sized,
{
    type Output = Result<Ndarr<T, { R - 1 }>, DimError>;
    fn reduce(self, axis: usize, f: F) -> Self::Output {
        if axis >= R {
            Err(DimError::new("Axis grater than rank"))
        } else {
            let slices = self.slice_at(axis);
            let n = slices.len();
            let mut out = slices[0].clone();
            for i in 1..n{
                out.bimap_in_place(slices[i].clone(), f.clone())
            }

            Ok(out)
        }
    }
}

pub trait Broadcast<const R2: usize>{
    type Output;
    fn broadcast_to(&self, shape: &[usize; R2]) -> Result<Self::Output, DimError>;//try broadcast to shape 
    fn broadcast(&self, shape: &[usize; R2]) -> Result<Self::Output, DimError>;//try broadcasting to compatible shape between self.shape and shape
}

impl <T, const R1: usize, const R2: usize> Broadcast<R2> for Ndarr<T, R1>
    where 
    T: Copy + Clone + Default + Debug,
    [usize; {const_max(R1, R2)}]: Sized
    
{
    type Output = Ndarr<T, {const_max(R1, R2)}>;
    fn broadcast_to(&self, shape: &[usize; R2]) -> Result<Self::Output,DimError> {
        //see https://numpy.org/doc/stable/user/basics.broadcasting.html
        //TODO: not sure at all if this implementation is general, but it seems to work for Rank 1 2 array broadcasted up to rank 3. For higher ranks a more rigorous proof is needed.
        let new_shape = helpers::broadcast_shape(&self.shape, shape)?;

        if new_shape.len() > R2{
            return Err(DimError { details: "Array can not be broadcasted to shape".to_string() });
        }else{
            let n_old = helpers::multiply_list(&self.shape, 1);
            let n = helpers::multiply_list(&new_shape, 1);
            let repetitions = n / n_old;

            let mut new_data = vec![T::default(); n];
            for i in 0..repetitions{
                for j in 0..n_old{
                    new_data[i*n_old + j] = self.data[j]
                }

            }

                return Ok(Ndarr{data: new_data, shape: new_shape})
        }
    }
    fn broadcast(&self, shape: &[usize; R2]) -> Result<Self::Output, DimError> {
        
        let new_shape = helpers::broadcast_shape(&self.shape, shape)?;

        let n_old = helpers::multiply_list(&self.shape, 1);
        let n = helpers::multiply_list(&new_shape, 1);
        let repetitions = n / n_old;

        let mut new_data = vec![T::default(); n];
        for i in 0..n{
            let indexes = get_indexes(&i, &new_shape);
            let rev_casted_pos = rev_cast_pos(&self.shape, &indexes)?;
            new_data[i] = self.data[rev_casted_pos];
        }
        Ok(Ndarr { data: new_data, shape: new_shape })

    }

}