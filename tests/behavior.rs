//! Golden-value integration tests for asymmetric shapes.

use rapl::*;

fn stack<T: Clone, R: Rank + InsertAxisRank>(
    arrays: &[Ndarr<T, R>],
    axis: usize,
) -> Ndarr<T, Inserted<R>> {
    let expanded: Vec<_> = arrays
        .iter()
        .map(|array| array.insert_axis_view(axis).unwrap())
        .collect();
    Ndarr::concatenate(axis, &expanded).unwrap()
}

/// Owned rank-reduced arrays along one axis.
fn owned_slices<T: Clone, R: Rank + RemoveAxisRank>(
    array: &Ndarr<T, R>,
    axis: usize,
) -> Vec<Ndarr<T, Removed<R>>> {
    (0..array.shape()[axis])
        .map(|index| array.index_axis_view(axis, index).unwrap().to_owned_array())
        .collect()
}

fn arr_2_3_4() -> Ndarr<i32, typenum::U3> {
    let data: Vec<i32> = (0..24).collect();
    Ndarr::new(&data, [2, 3, 4]).unwrap()
}

#[test]
fn transpose_asymmetric() {
    let a = arr_2_3_4();
    let t = a.t();
    assert_eq!(t.shape(), &[4, 3, 2]);
    assert_eq!(
        t.data(),
        vec![0, 12, 4, 16, 8, 20, 1, 13, 5, 17, 9, 21, 2, 14, 6, 18, 10, 22, 3, 15, 7, 19, 11, 23]
    );
}

#[test]
fn slice_asymmetric() {
    let a = arr_2_3_4();

    let s0 = owned_slices(&a, 0);
    assert_eq!(s0.len(), 2);
    assert_eq!(s0[0].shape(), &[3, 4]);
    assert_eq!(s0[0].data(), (0..12).collect::<Vec<_>>());
    assert_eq!(s0[1].data(), (12..24).collect::<Vec<_>>());

    let s1 = owned_slices(&a, 1);
    assert_eq!(s1.len(), 3);
    assert_eq!(s1[0].shape(), &[2, 4]);
    assert_eq!(s1[0].data(), vec![0, 1, 2, 3, 12, 13, 14, 15]);
    assert_eq!(s1[1].data(), vec![4, 5, 6, 7, 16, 17, 18, 19]);

    let s2 = owned_slices(&a, 2);
    assert_eq!(s2.len(), 4);
    assert_eq!(s2[0].shape(), &[2, 3]);
    assert_eq!(s2[0].data(), vec![0, 4, 8, 12, 16, 20]);
    assert_eq!(s2[1].data(), vec![1, 5, 9, 13, 17, 21]);
}

#[test]
fn reduce_asymmetric() {
    let a = arr_2_3_4();
    let r0 = a.reduce(0, |x, y| x + y).unwrap();
    assert_eq!(r0.shape(), &[3, 4]);
    assert_eq!(
        r0.data(),
        vec![12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34]
    );

    let r1 = a.reduce(1, |x, y| x + y).unwrap();
    assert_eq!(r1.shape(), &[2, 4]);
    assert_eq!(r1.data(), vec![12, 15, 18, 21, 48, 51, 54, 57]);

    let r2 = a.reduce(2, |x, y| x + y).unwrap();
    assert_eq!(r2.shape(), &[2, 3]);
    assert_eq!(r2.data(), vec![6, 22, 38, 54, 70, 86]);
}

#[test]
fn reshape_asymmetric() {
    let a = arr_2_3_4();
    let r = a.reshape([4, 6]).unwrap();
    assert_eq!(r.shape(), &[4, 6]);
    assert_eq!(r.data(), (0..24).collect::<Vec<_>>());
}

#[test]
fn broadcast_add_asymmetric() {
    let b = Ndarr::from([1, 2, 3, 4]);
    let c = Ndarr::from([[10, 20, 30, 40], [50, 60, 70, 80]]);
    assert_eq!((&b + &c).data(), vec![11, 22, 33, 44, 51, 62, 73, 84]);
}

#[test]
fn matmul_golden() {
    let m1 = Ndarr::from([[1, 2, 3], [4, 5, 6]]);
    let m2 = Ndarr::from([[7, 8], [9, 10], [11, 12]]);
    assert_eq!(m1.mat_mul(&m2).unwrap().data(), vec![58, 64, 139, 154]);
}

#[test]
fn stack_roundtrip_asymmetric() {
    let a = arr_2_3_4();
    for axis in 0..3 {
        let slices = owned_slices(&a, axis);
        assert_eq!(stack(&slices, axis), a);
    }
}

#[test]
fn roll_asymmetric() {
    let a = arr_2_3_4();
    let rolled = a.roll(1, 0);
    // Rolling axis 0 by 1: second block moves to front
    assert_eq!(
        owned_slices(&rolled, 0)[0].data(),
        (12..24).collect::<Vec<_>>()
    );
    assert_eq!(
        owned_slices(&rolled, 0)[1].data(),
        (0..12).collect::<Vec<_>>()
    );
}

/// `roll` works on any view: rolling a view equals rolling its materialization.
#[test]
fn roll_applies_to_views() {
    let a = arr_2_3_4();
    for axis in 0..3 {
        for shift in [-2isize, -1, 0, 1, 5] {
            assert_eq!(
                a.t_view().roll(shift, axis),
                a.t().roll(shift, axis),
                "axis={axis} shift={shift}"
            );
        }
    }
}

#[test]
fn views_match_eager_ops() {
    let a = arr_2_3_4();
    assert_eq!(a.t_view().to_owned_array(), a.t());
    assert_eq!(
        a.index_axis_view(1, 0).unwrap().to_owned_array(),
        owned_slices(&a, 1)[0]
    );
    let target = Dim::<typenum::U3>::new(&[2, 3, 4]).unwrap();
    let row = Ndarr::from([10, 20, 30, 40]);
    let bview = row.broadcast_view_to(&target).unwrap();
    assert_eq!(bview.shape(), &[2, 3, 4]);
    assert_eq!(bview[[0, 0, 0]], 10);
    assert_eq!(bview[[0, 0, 3]], 40);
    assert_eq!(bview[[0, 1, 0]], 10);
}

#[test]
fn mutable_axis_view() {
    let mut a = Ndarr::from([[1, 2], [3, 4]]);
    {
        let mut row = a.index_axis_mut(0, 1).unwrap();
        row[[0]] = 30;
        row[[1]] = 40;
    }
    assert_eq!(a, Ndarr::from([[1, 2], [30, 40]]));
}

#[test]
fn map_view_does_not_touch_siblings() {
    let mut a = Ndarr::from([[1, 2], [3, 4]]);
    {
        let mut row = a.index_axis_mut(0, 0).unwrap();
        row.map_in_place(|x| x + 10);
    }
    assert_eq!(a, Ndarr::from([[11, 12], [3, 4]]));

    let a = Ndarr::from([[1, 2], [3, 4]]);
    let mapped = a.index_axis_view(0, 0).unwrap().map(|x| x * 2);
    assert_eq!(mapped, Ndarr::from([2, 4]));
    assert_eq!(mapped.len(), 2);
}

#[test]
fn from_str_multibyte_shape_matches_char_count() {
    let a = Ndarr::from("héllo");
    assert_eq!(a.shape(), &[5]);
    assert_eq!(a.data(), vec!['h', 'é', 'l', 'l', 'o']);
}

#[test]
fn zip_with_in_place_rejects_shapes_that_do_not_broadcast() {
    let mut a = Ndarr::from([1, 2, 3]);
    assert!(a
        .zip_with_in_place(&Ndarr::from([1, 2]), |x, y| x + y)
        .is_err());
    // Broadcasting runs one way: `other` must fit `self`'s shape.
    let wide = Ndarr::from([[1, 2, 3], [4, 5, 6]]);
    assert!(a.zip_with_in_place(&wide, |x, y| x + y).is_err());
    assert_eq!(a, Ndarr::from([1, 2, 3]));
}

#[test]
#[should_panic(expected = "must broadcast")]
fn assign_operator_panics_when_the_right_side_does_not_broadcast() {
    let mut a = Ndarr::from([1, 2, 3]);
    a += &Ndarr::from([1, 2]);
}

#[test]
fn debug_output_hides_view_metadata() {
    let a = Ndarr::from([[1, 2, 3], [4, 5, 6]]);
    let debug = format!("{:?}", a);
    assert!(debug.starts_with("Ndarr"), "got: {debug}");
    assert!(debug.contains("data"), "got: {debug}");
    assert!(debug.contains("dim"), "got: {debug}");
    assert!(!debug.contains("offset"), "got: {debug}");
    assert!(!debug.contains("strides"), "got: {debug}");
    assert!(!debug.contains("buffer"), "got: {debug}");
}

#[test]
fn reshape_view_contiguous() {
    let a = arr_2_3_4();
    let v = a.view().reshape([4, 6]).unwrap();
    assert_eq!(v.shape(), &[4, 6]);
    assert_eq!(v.to_owned_array(), a.reshape([4, 6]).unwrap());
}

#[test]
fn zip_with_in_place_broadcasts_other_shapes_and_element_types() {
    let mut grid = Ndarr::from([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    grid.zip_with_in_place(&Ndarr::from([10.0, 20.0, 30.0]), |x, y| x + y)
        .unwrap(); // row, broadcast down the columns
    grid.zip_with_in_place(&Ndarr::from([[100.0], [200.0]]), |x, y| x + y)
        .unwrap(); // column, broadcast across the rows
    assert_eq!(
        grid,
        Ndarr::from([[111.0, 122.0, 133.0], [214.0, 225.0, 236.0]])
    );

    let hide = Ndarr::from_fn([2, 3], |ix| ix[1] > ix[0]); // strictly above the diagonal
    grid.zip_with_in_place(&hide, |&x, &hidden| if hidden { 0.0 } else { x })
        .unwrap();
    assert_eq!(grid, Ndarr::from([[111.0, 0.0, 0.0], [214.0, 225.0, 0.0]]));

    let scalar = Ndarr::<f64, rapl::U0>::from_fn([], |_| 0.5);
    grid.zip_with_in_place(&scalar, |x, y| x * y).unwrap();
    assert_eq!(grid[[1, 1]], 112.5);
}

#[test]
fn zip_with_in_place_updates_strided_and_permuted_targets() {
    let mut a = Ndarr::from([[1, 2, 3], [4, 5, 6]]);
    a.slice_mut(s![.., ..;-1])
        .unwrap()
        .zip_with_in_place(&Ndarr::from([100, 200, 300]), |x, y| x + y)
        .unwrap();
    assert_eq!(a, Ndarr::from([[301, 202, 103], [304, 205, 106]]));

    let mut permuted = a.permute_axes(&[1, 0]).unwrap(); // [3, 2], not C order
    permuted
        .zip_with_in_place(&Ndarr::from([1, -1]), |x, y| x * y)
        .unwrap();
    assert_eq!(
        permuted,
        Ndarr::from([[301, -304], [202, -205], [103, -106]])
    );
}

#[test]
fn assign_operators_broadcast_the_right_side() {
    let mut activations = Ndarr::<i32, rapl::U3>::zeros([2, 2, 3]);
    activations += &Ndarr::from([1, 2, 3]); // bias per feature
    activations *= &Ndarr::from([[1], [10]]);
    assert_eq!(
        activations,
        Ndarr::from([[[1, 2, 3], [10, 20, 30]], [[1, 2, 3], [10, 20, 30]]])
    );
}
