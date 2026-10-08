//! Owned arrays may carry permuted axes: `permute_axes` and `reshape` consume
//! and keep storage, `into_data` restores logical order, and raw `data` access
//! refuses a layout that is not C order.

use rapl::{Dyn, Ndarr, U2, U3};

fn asymmetric() -> Ndarr<i32, U3> {
    Ndarr::new(&(0..24).collect::<Vec<_>>(), [2, 3, 4]).unwrap()
}

#[test]
fn permute_and_reshape_keep_the_owned_allocation() {
    let array = asymmetric();
    let allocation = array.data().as_ptr();
    let permuted = array.permute_axes(&[2, 0, 1]).unwrap();
    assert!(!permuted.is_standard_layout());
    let restored = permuted.permute_axes(&[1, 2, 0]).unwrap();
    assert!(restored.is_standard_layout());
    let flat: Ndarr<i32, U2> = restored.reshape([6, 4]).unwrap();
    assert_eq!(flat.data().as_ptr(), allocation);
    assert_eq!(flat.data(), (0..24).collect::<Vec<_>>());
}

#[test]
fn permuted_owned_arrays_behave_like_permuted_views() {
    let array = asymmetric();
    let expected = array.view().permute_axes(&[2, 0, 1]).unwrap();
    let permuted = array.clone().permute_axes(&[2, 0, 1]).unwrap();
    assert_eq!(permuted, expected);
    assert_eq!(permuted.shape(), &[4, 2, 3]);
    assert_eq!(permuted[[3, 1, 2]], array[[1, 2, 3]]);
    assert_eq!(
        permuted.iter_elems().collect::<Vec<_>>(),
        expected.iter_elems().collect::<Vec<_>>()
    );
    assert_eq!(permuted.map(|v| v * 2), expected.map(|v| v * 2));
    assert_eq!(&permuted + 1, &expected + 1);
    assert_eq!(
        format!("{permuted}"),
        format!("{}", expected.to_owned_array())
    );
    assert_eq!(permuted.clone().into_dyn(), expected);
    assert_eq!(
        permuted.clone().into_dyn().into_ranked::<U3>().unwrap(),
        expected
    );

    let mut mutated = permuted.clone();
    mutated[[0, 0, 0]] = 100;
    mutated.map_in_place(|v| v + 1);
    mutated += 1;
    assert_eq!(mutated[[0, 0, 0]], 102);
    assert_eq!(mutated[[3, 1, 2]], array[[1, 2, 3]] + 2);
}

#[test]
fn into_data_and_to_owned_array_restore_logical_order() {
    let permuted = asymmetric().permute_axes(&[1, 2, 0]).unwrap();
    let logical: Vec<i32> = permuted.iter_elems().copied().collect();
    let standard = permuted.to_owned_array();
    assert!(standard.is_standard_layout());
    assert_eq!(standard.data(), logical);
    assert_eq!(permuted.into_data(), logical);

    // Reordering moves elements; it never needs `Clone`.
    struct NotClone(usize);
    let mut next = 0;
    let owned = Ndarr::<NotClone, U2>::from_fn([2, 3], |_| {
        next += 1;
        NotClone(next)
    });
    let transposed = owned.permute_axes(&[1, 0]).unwrap();
    let values: Vec<usize> = transposed.into_data().into_iter().map(|v| v.0).collect();
    assert_eq!(values, vec![1, 4, 2, 5, 3, 6]);
}

#[test]
fn raw_data_and_reshape_refuse_a_permuted_layout() {
    let permuted = asymmetric().permute_axes(&[2, 1, 0]).unwrap();
    assert!(std::panic::catch_unwind(|| permuted.data().len()).is_err());
    let mut mutable = permuted.clone();
    let result = std::panic::catch_unwind(move || mutable.data_mut().len());
    assert!(result.is_err());
    assert!(permuted.clone().reshape([24]).is_err());
    let flat: Ndarr<i32, Dyn> = permuted.to_owned_array().reshape(vec![24]).unwrap();
    assert_eq!(flat.shape(), &[24]);
}

#[test]
fn unit_and_empty_axes_do_not_make_a_layout_permuted() {
    let unit = Ndarr::<i32, U3>::new(&[1, 2, 3], [1, 3, 1])
        .unwrap()
        .permute_axes(&[2, 1, 0])
        .unwrap();
    assert!(unit.is_standard_layout());
    assert_eq!(unit.data(), &[1, 2, 3]);
    let empty = Ndarr::<i32, U3>::new(&[], [2, 0, 3])
        .unwrap()
        .permute_axes(&[2, 0, 1])
        .unwrap();
    assert!(empty.data().is_empty());
    assert!(empty.into_data().is_empty());
}
