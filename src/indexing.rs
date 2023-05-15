use std::ops::{Index, IndexMut};

use super::*;

impl<T: Clone + Debug + Default, R: Unsigned> Ndarr<T, R> {
    ///Modifies an element at an specific position by assign a new value.
    pub fn assign_at<D: Into<Dim<R>>>(&mut self, index: D, value: T) {
        let flat_pos = self.dim.get_flat_pos(&index.into()).unwrap();
        self.data[flat_pos] = value;
    }
}


fn slice_borrow<'a, T: Clone + Debug + Default, R: Unsigned>(arr: &'a Ndarr<T,R>, axis: usize)->Vec<Ndarr<&'a T,UTerm>>{
        let n = helpers::multiply_list(&arr.dim.shape, 1); // number of elements in original array
        let new_shape = arr.dim.clone().remove_element_notyped(axis);
        let n_new_arrs = arr.dim.shape[axis]; // number of new arrays

        let iota = 0..n;

        let indexes: Vec<Dim<R>> = iota.map(|i| arr.dim.get_indexes(&i)).collect(); //indexes of each element

        let mut out:Vec<Ndarr<&'a T, UTerm>> = Vec::new(); // to store

        for i in 0..n_new_arrs {
            let mut this_data: Vec<&'a T> = Vec::new();
            for j in 0..n {
                // if the index at the slice position coincide with i (or the slice number)
                if indexes[j].shape[axis] == i {
                    let ind = arr.dim.get_flat_pos(&indexes[j]).unwrap();
                    this_data.push(&arr.data[ind])
                }
            }
            //TODO: remove push, with allocation size
            out.push(Ndarr::new(&this_data, new_shape.clone()).expect("Error initializing"))
        }
        out
}




impl<T: Clone + Default + Debug, R: Unsigned, I: Into<Dim<R>>> Index<I> for Ndarr<T, R> {
    type Output = T;
    fn index(&self, index: I) -> &Self::Output {
        let flat_pos = self.dim.get_flat_pos(&index.into()).unwrap();
        &self.data[flat_pos]
    }
}

impl<T: Clone + Default + Debug, R: Unsigned, I: Into<Dim<R>>> IndexMut<I> for Ndarr<T, R> {
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        let flat_pos = self.dim.get_flat_pos(&index.into()).unwrap();
        &mut self.data[flat_pos]
    }
}

#[cfg(test)]
mod indexing_tes {
    use super::*;
    #[test]
    fn assing_at_t() {
        let mut arr = Ndarr::from([[1, 2], [3, 4]]);
        arr.assign_at([0, 1], 8);
        assert_eq!(&arr, &Ndarr::from([[1, 8], [3, 4]]));
        arr.assign_at([1, 1], 10);
        assert_eq!(&arr, &Ndarr::from([[1, 8], [3, 10]]));
    }

    #[test]
    fn indexing() {
        let mut arr = Ndarr::from([[1, 2], [3, 4]]);
        assert_eq!(arr[[0, 0]], 1);
        arr[[1, 1]] = 10;
        assert_eq!(&arr, &Ndarr::from([[1, 2], [3, 10]]));
    }
}
