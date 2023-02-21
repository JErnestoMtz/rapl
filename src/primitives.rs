use super::*;
use crate::helpers::*;

#[derive(Debug)]
struct DimError {
    details: String,
}

impl DimError {
    fn new(msg: &str) -> DimError {
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
    T: Copy + Clone + Default,
    [usize; R]: Default,
    [usize;  R - 1 ]: Sized,
{
    type Output = Vec<Ndarr<T, { R - 1 }>>;
    fn slice_at(self, axis: usize) -> Self::Output {
        let n = multiply_list(&self.shape, 1); // number of elements in original array
        let new_shape: [usize;  R - 1 ] = remove_element(self.shape, axis);
        let new_n = multiply_list(&new_shape, 1); // number of elements in each splitted array
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
            out.push(Ndarr {
                data: this_data,
                shape: new_shape,
            })
        }
        out
    }
}
trait Reduce<F> {
    type Output;
    fn reduce(self, axis: usize, f: F) -> Self::Output;
}

impl<T, F, const R: usize> Reduce<F> for Ndarr<T, R>
where
    F: Fn(T, T) -> T,
    T: Copy + Clone + Default,
    [usize;  R - 1 ]: Sized,
{
    type Output = Result<Ndarr<T, { R - 1 }>, DimError>;
    fn reduce(self, axis: usize, f: F) -> Self::Output {
        if axis >= R {
            Err(DimError::new("Axis grater than rank"))
        } else {
            let new_shape: [usize;  R - 1 ] = remove_element(self.shape, axis);
            todo!()
        }
    }
}
