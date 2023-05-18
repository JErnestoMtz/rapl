use std::ops::{Index, IndexMut};
use super::*;

impl<T: Clone + Debug, R: Unsigned> Ndarr<T, R> {
    ///Modifies an element at an specific position by assign a new value.
    pub fn assign_at<D: Into<Dim<R>>>(&mut self, index: D, value: T) {
        let flat_pos = self.dim.get_flat_pos(&index.into()).unwrap();
        self.data[flat_pos] = value;
    }
    pub fn index_slice_notyped(&self, axis: usize, index: usize) -> Ndarr<T, UTerm>{
        let new_shape = self.dim.clone().remove_element_notyped(axis);
        let n_new = helpers::multiply_list(&new_shape.shape, 1); // number of elements in new slice;

        let iota = 0..n_new;
        let indexes: Vec<Dim<UTerm>> = iota.map(|i| new_shape.get_indexes(&i).insert_element_notyped(axis, index)).collect(); //indexes of each element
        let flat_pos: Vec<usize> = indexes.iter().map(|index| self.dim.get_flat_pos(index).unwrap()).collect();
        let new_data: Vec<T> = flat_pos.iter().map(|i| self.data[*i].clone()).collect();
        return Ndarr::new(&new_data, new_shape).unwrap()
    }
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


#[cfg(test)]

mod indexing_test{
    use super::*;
    #[test]
    fn index_slice(){
        let a = Ndarr::from([[1,2,3],[4,5,6]]);
        let b = a.index_slice_notyped(0, 1);
        assert_eq!(b.data, vec![4,5,6])
        
    }
}