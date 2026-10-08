use rapl::{s, Dim, Dyn, Ndarr, U2, U3};

#[test]
fn concatenate_joins_existing_axes_with_different_extents() {
    let top = Ndarr::from([[1, 2], [3, 4]]);
    let bottom = Ndarr::from([[5, 6]]);
    assert_eq!(
        Ndarr::concatenate(0, &[top.view(), bottom.view()]).unwrap(),
        Ndarr::from([[1, 2], [3, 4], [5, 6]])
    );

    let left = Ndarr::from([[1, 2], [3, 4]]);
    let right = Ndarr::from([[5], [6]]);
    assert_eq!(
        Ndarr::concatenate(1, &[left.view(), right.view()]).unwrap(),
        Ndarr::from([[1, 2, 5], [3, 4, 6]])
    );
}

#[test]
// NumPy-style descending bounds (`3..1;-1` selects 3, 2) are literal ranges
// clippy reads as empty.
#[allow(clippy::reversed_empty_ranges)]
fn concatenate_reads_strided_and_broadcast_views_in_logical_order() {
    let array = Ndarr::from([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]);
    let left = array.slice(s![.., 0..2]).unwrap();
    let right = array.slice(s![.., 3..1;-1]).unwrap();
    assert_eq!(
        Ndarr::concatenate(1, &[left, right]).unwrap(),
        Ndarr::from([[0, 1, 3, 2], [4, 5, 7, 6], [8, 9, 11, 10]])
    );

    let row = Ndarr::from([1, 2, 3]);
    let two = row
        .broadcast_view_to(&Dim::<U2>::new(&[2, 3]).unwrap())
        .unwrap();
    let one = row
        .broadcast_view_to(&Dim::<U2>::new(&[1, 3]).unwrap())
        .unwrap();
    assert_eq!(
        Ndarr::concatenate(0, &[two, one]).unwrap(),
        Ndarr::from([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
    );
}

#[test]
fn stack_is_axis_insertion_followed_by_concatenation() {
    let a = Ndarr::from([[1, 2], [3, 4]]);
    let b = Ndarr::from([[5, 6], [7, 8]]);
    let expanded = [
        a.insert_axis_view(2).unwrap(),
        b.insert_axis_view(2).unwrap(),
    ];
    assert_eq!(
        Ndarr::concatenate(2, &expanded).unwrap(),
        Ndarr::from([[[1, 5], [2, 6]], [[3, 7], [4, 8]]])
    );
}

#[test]
fn concatenate_handles_zero_extents() {
    let empty = Ndarr::<i32, U3>::new(&[], [2, 0, 3]).unwrap();
    let values = Ndarr::from([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]);
    assert_eq!(
        Ndarr::concatenate(1, &[empty.view(), values.view()]).unwrap(),
        values
    );
}

#[test]
fn concatenate_rejects_missing_or_incompatible_inputs() {
    let none: [Ndarr<i32, U2>; 0] = [];
    assert!(Ndarr::concatenate(0, &none).is_err());

    let a = Ndarr::from([[1, 2], [3, 4]]);
    let wrong = Ndarr::from([[1, 2, 3]]);
    assert!(Ndarr::concatenate(0, &[a.view(), wrong.view()]).is_err());
    assert!(Ndarr::concatenate(2, &[a.view()]).is_err());

    let rank_two = Ndarr::<i32, Dyn>::new(&[1, 2], Dim::new(&[1, 2]).unwrap()).unwrap();
    let rank_one = Ndarr::<i32, Dyn>::new(&[3, 4], Dim::new(&[2]).unwrap()).unwrap();
    assert!(Ndarr::concatenate(0, &[rank_two, rank_one]).is_err());
}
