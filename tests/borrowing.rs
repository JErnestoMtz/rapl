//! Read-only primitives borrow inputs even when elements cannot be cloned.
use rapl::{s, Dim, Ndarr, Win, U0, U1, U2};
use std::cell::Cell;
use std::fmt;

#[derive(Debug, PartialEq, Eq)]
struct Value(i32);
impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

#[test]
fn nonclone_elements_compose_across_views() {
    let mut a = Ndarr::from_vec_dim((1..=6).map(Value).collect(), Dim::<U2>::from([2, 3])).unwrap();
    let b = Ndarr::from_vec_dim((1..=6).map(Value).collect(), Dim::<U2>::from([2, 3])).unwrap();
    assert_eq!(a, b);
    fn requires_eq<T: Eq>(_: &T) {}
    requires_eq(&a);
    assert!(format!("{a:?}").contains("Value(1)"));
    assert!(format!("{a}").contains('6'));
    assert!(std::ptr::eq(a.iter_elems().next().unwrap(), &a[[0, 0]]));
    assert_eq!(
        a.t_view().iter_elems().map(|v| v.0).collect::<Vec<_>>(),
        [1, 4, 2, 5, 3, 6]
    );
    let reversed = a.slice(s![.., ..;-1]).unwrap();
    assert_eq!(reversed.map(|v| v.0).data(), [3, 2, 1, 6, 5, 4]);
    let row = a.slice(s![0, ..]).unwrap();
    let repeated = row.broadcast_view_to(&Dim::<U2>::from([2, 3])).unwrap();
    assert_eq!(
        a.zip_with(&repeated, |x, y| Value(x.0 + y.0))
            .unwrap()
            .map(|v| v.0)
            .data(),
        [2, 4, 6, 5, 7, 9]
    );
    assert_eq!(
        a.fold_axis(1, 0, |acc, v| acc + v.0).unwrap().data(),
        [6, 15]
    );
    let windows = row.slice(s![Win(2)]).unwrap();
    let kernel = Ndarr::from_vec_dim(vec![Value(2), Value(3)], Dim::<U1>::from([2])).unwrap();
    let result = windows
        .contract(
            &kernel,
            U1::new(),
            |x, y| Value(x.0 * y.0),
            |x, y| Value(x.0 + y.0),
        )
        .unwrap();
    assert_eq!(result.map(|v| v.0).data(), [8, 13]);
    a.slice_mut(s![.., ..;-1])
        .unwrap()
        .zip_with_in_place(&b, |x, y| Value(x.0 + y.0))
        .unwrap();
    assert_eq!(a.map(|v| v.0).data(), [4, 4, 4, 10, 10, 10]);
}

#[test]
fn borrowing_handles_scalar_and_empty_arrays() {
    let scalar = Ndarr::from_vec_dim(vec![Value(7)], Dim::<U0>::from([])).unwrap();
    assert_eq!(scalar.iter_elems().count(), 1);
    assert_eq!(scalar.map(|v| v.0).scalar(), 7);
    let empty = Ndarr::from_vec_dim(Vec::<Value>::new(), Dim::<U2>::from([2, 0])).unwrap();
    assert_eq!(empty.iter_elems().size_hint(), (0, Some(0)));
    assert!(empty.map(|_| panic!("empty map")).is_empty());
    assert_eq!(
        empty
            .fold_axis(1, 42, |_, _| panic!("empty fold"))
            .unwrap()
            .data(),
        [42, 42]
    );
    let out = empty
        .outer_product(&scalar, |_, _| -> Value { panic!("empty product") })
        .unwrap();
    assert_eq!(out.shape(), [2, 0]);
}

#[test]
fn borrowing_primitives_never_clone_inputs() {
    struct Counted<'a>(&'a Cell<usize>, i32);
    impl Clone for Counted<'_> {
        fn clone(&self) -> Self {
            self.0.set(self.0.get() + 1);
            Self(self.0, self.1)
        }
    }
    let clones = Cell::new(0);
    let mut a = Ndarr::from_vec_dim(
        vec![Counted(&clones, 2), Counted(&clones, 3)],
        Dim::<U1>::from([2]),
    )
    .unwrap();
    let b = Ndarr::from_vec_dim(
        vec![Counted(&clones, 4), Counted(&clones, 5)],
        Dim::<U1>::from([2]),
    )
    .unwrap();
    assert_eq!(a.iter_elems().map(|x| x.1).sum::<i32>(), 5);
    assert_eq!(a.map(|x| x.1).data(), [2, 3]);
    assert_eq!(a.zip_with(&b, |x, y| x.1 + y.1).unwrap().data(), [6, 8]);
    assert_eq!(a.fold_axis(0, 0, |acc, x| acc + x.1).unwrap().scalar(), 5);
    assert_eq!(
        a.contract(&b, U1::new(), |x, y| x.1 * y.1, |x, y| x + y)
            .unwrap()
            .scalar(),
        23
    );
    a.zip_with_in_place(&b, |x, y| Counted(x.0, x.1 + y.1))
        .unwrap();
    assert_eq!(clones.get(), 0);
    let owned = a.to_owned_array();
    assert_eq!(clones.get(), 2);
    assert_eq!(owned.map(|x| x.1).data(), [6, 8]);
}
