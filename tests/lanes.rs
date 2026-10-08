use rapl::{
    s, BroadcastRank, Dim, Dyn, InsertAxisRank, Keep, Ndarr, Rank, RemoveAxisRank, Removed,
    ScanDirection, U1, U2, U3,
};

fn asymmetric() -> Ndarr<i32, U3> {
    Ndarr::new(&(0..24).collect::<Vec<_>>(), [2, 3, 4]).unwrap()
}

#[test]
fn lanes_are_rank_one_views_in_logical_order() {
    let array = asymmetric();
    let lanes: Vec<_> = array
        .lanes(1)
        .unwrap()
        .map(|lane| {
            assert_eq!(lane.shape(), &[3]);
            assert_eq!(lane.strides(), &[4]);
            lane.iter_elems().cloned().collect::<Vec<_>>()
        })
        .collect();

    assert_eq!(
        lanes,
        vec![
            vec![0, 4, 8],
            vec![1, 5, 9],
            vec![2, 6, 10],
            vec![3, 7, 11],
            vec![12, 16, 20],
            vec![13, 17, 21],
            vec![14, 18, 22],
            vec![15, 19, 23],
        ]
    );
}

#[test]
fn lanes_honor_transpose_negative_steps_and_broadcast_strides() {
    let array = Ndarr::from([[0, 1, 2], [3, 4, 5]]);
    let transposed: Vec<Vec<_>> = array
        .t_view()
        .lanes(1)
        .unwrap()
        .map(|lane| lane.iter_elems().cloned().collect())
        .collect();
    assert_eq!(transposed, vec![vec![0, 3], vec![1, 4], vec![2, 5]]);

    let reversed = array.slice(s![.., ..;-1]).unwrap();
    let reversed: Vec<Vec<_>> = reversed
        .lanes(1)
        .unwrap()
        .map(|lane| lane.iter_elems().cloned().collect())
        .collect();
    assert_eq!(reversed, vec![vec![2, 1, 0], vec![5, 4, 3]]);

    let row = Ndarr::from([1, 2, 3]);
    let broadcast = row
        .broadcast_view_to(&Dim::<U2>::new(&[2, 3]).unwrap())
        .unwrap();
    let broadcast: Vec<Vec<_>> = broadcast
        .lanes(0)
        .unwrap()
        .map(|lane| lane.iter_elems().cloned().collect())
        .collect();
    assert_eq!(broadcast, vec![vec![1, 1], vec![2, 2], vec![3, 3]]);
}

#[test]
fn dynamic_arrays_produce_the_same_fixed_rank_lanes() {
    let array = Ndarr::<i32, Dyn>::new(
        &(0..6).collect::<Vec<_>>(),
        Dim::<Dyn>::new(&[2, 3]).unwrap(),
    )
    .unwrap();
    let lanes: Vec<Ndarr<i32, U1>> = array
        .lanes(0)
        .unwrap()
        .map(|lane| lane.to_owned_array())
        .collect();
    assert_eq!(
        lanes,
        vec![
            Ndarr::from([0, 3]),
            Ndarr::from([1, 4]),
            Ndarr::from([2, 5]),
        ]
    );
}

#[test]
fn fold_axis_uses_one_seed_for_each_lane_including_empty_lanes() {
    let array = asymmetric();
    assert_eq!(
        array.fold_axis(1, 0, |acc, value| acc + value).unwrap(),
        Ndarr::from([[12, 15, 18, 21], [48, 51, 54, 57]])
    );

    let empty = Ndarr::<i32, U3>::new(&[], [2, 0, 3]).unwrap();
    let folded = empty.fold_axis(1, 7, |acc, value| acc + value).unwrap();
    assert_eq!(folded.shape(), &[2, 3]);
    assert_eq!(folded.data(), &[7; 6]);

    let no_lanes = empty.fold_axis(0, 7, |acc, value| acc + value).unwrap();
    assert_eq!(no_lanes.shape(), &[0, 3]);
    assert!(no_lanes.data().is_empty());
}

#[test]
fn map_lanes_assembles_non_contiguous_lane_results_in_c_order() {
    let array = asymmetric();
    let reversed = array
        .map_lanes(1, |lane| {
            let mut values: Vec<_> = lane.iter_elems().cloned().collect();
            values.reverse();
            Ndarr::new(&values, [values.len()]).unwrap()
        })
        .unwrap();
    assert_eq!(
        reversed.data(),
        &[8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3, 20, 21, 22, 23, 16, 17, 18, 19, 12, 13, 14, 15,]
    );

    assert!(array.map_lanes(1, |_lane| Ndarr::from([1, 2])).is_err());
    assert!(array.lanes(3).is_err());
}

#[test]
fn explicit_scans_define_direction() {
    let array = Ndarr::from([1, 2, 3]);
    assert_eq!(
        array
            .scan_axis(0, ScanDirection::Forward, |acc, value| acc - value,)
            .unwrap(),
        Ndarr::from([1, -1, -4])
    );
    assert_eq!(
        array
            .scan_axis(0, ScanDirection::Backward, |acc, value| acc - value,)
            .unwrap(),
        Ndarr::from([0, 1, 3])
    );
    // Non-commutative combiner: swapping accumulator and element flips the result.
    assert_eq!(
        array
            .scan_axis(0, ScanDirection::Forward, |acc, value| value - acc)
            .unwrap(),
        Ndarr::from([1, 1, 2])
    );
    assert_eq!(
        array
            .scan_axis(0, ScanDirection::Backward, |acc, value| value - acc)
            .unwrap(),
        Ndarr::from([2, -1, 3])
    );
}

/// The reduction family spelled with primitives: `fold_axis` (seeded), `reduce` (seedless), std adapters (global).
#[test]
fn fold_and_reduce_spell_the_reduction_family() {
    let array = Ndarr::from([[1, 2, 3], [4, 5, 6]]);
    assert_eq!(
        array.fold_axis(1, 0, |acc, value| acc + value).unwrap(),
        Ndarr::from([6, 15])
    );
    assert_eq!(
        array.fold_axis(0, 1, |acc, value| acc * value).unwrap(),
        Ndarr::from([4, 10, 18])
    );
    assert_eq!(array.reduce(1, i32::max).unwrap(), Ndarr::from([3, 6]));
    assert_eq!(array.reduce(0, i32::min).unwrap(), Ndarr::from([1, 2, 3]));
    assert_eq!(array.iter_elems().sum::<i32>(), 21);
    assert_eq!(array.iter_elems().cloned().max(), Some(6));

    let mask = Ndarr::<bool, U2>::new(&[true, false, true, true], [2, 2]).unwrap();
    assert_eq!(
        mask.fold_axis(0, false, |acc, value| acc || *value)
            .unwrap()
            .data(),
        &[true, true]
    );
    assert_eq!(
        mask.fold_axis(0, true, |acc, value| acc && *value)
            .unwrap()
            .data(),
        &[true, false]
    );

    // seeded folds of an empty axis return the seed; seedless folds error
    let empty = Ndarr::<i32, U2>::new(&[], [2, 0]).unwrap();
    assert_eq!(
        empty.fold_axis(1, 0, |acc, value| acc + value).unwrap(),
        Ndarr::from([0, 0])
    );
    assert_eq!(
        empty.fold_axis(1, 1, |acc, value| acc * value).unwrap(),
        Ndarr::from([1, 1])
    );
    assert!(empty.reduce(1, i32::max).is_err());
}

/// Softmax as evidence that reduce, axis insertion, and broadcasting compose.
fn softmax_axis<R>(array: &Ndarr<f64, R>, axis: usize) -> Ndarr<f64, R>
where
    R: Rank + RemoveAxisRank + BroadcastRank<R, Output = R>,
    Removed<R>: InsertAxisRank<Output = R>,
{
    let maxima = array.reduce(axis, f64::max).unwrap();
    let maxima = maxima.insert_axis_view(axis).unwrap();
    let exp = (array - &maxima).exp();
    let totals = exp.fold_axis(axis, 0.0, |acc, value| acc + value).unwrap();
    let totals = totals.insert_axis_view(axis).unwrap();
    &exp / &totals
}

#[test]
fn softmax_is_a_composition_over_the_primitives() {
    let array = Ndarr::from([[1.0_f64, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    let rows = softmax_axis(&array, 1);
    assert!(rows
        .fold_axis(1, 0.0, |acc, value| acc + value)
        .unwrap()
        .approx_epsilon(&Ndarr::from([1.0, 1.0]), 1e-8));

    let columns = softmax_axis(&array, 0);
    assert!(columns
        .fold_axis(0, 0.0, |acc, value| acc + value)
        .unwrap()
        .approx_epsilon(&Ndarr::from([1.0, 1.0, 1.0]), 1e-8));

    // Global softmax is the same operation over one flattened lane.
    let flat = array.clone().reshape([array.len()]).unwrap();
    let global = softmax_axis(&flat, 0).reshape(array.dim().clone()).unwrap();
    assert!((global.iter_elems().sum::<f64>() - 1.0).abs() < 1e-12);
}

#[test]
fn keep_matches_reduce_then_insert_axis_for_every_axis() {
    let array = asymmetric();
    for axis in 0..3 {
        let kept: Ndarr<i32, U3> = array.reduce(Keep(axis), |a, b| a + b).unwrap();
        let reinserted = array
            .reduce(axis, |a, b| a + b)
            .unwrap()
            .insert_axis_view(axis)
            .unwrap()
            .to_owned_array();
        assert_eq!(kept, reinserted, "reduce axis {axis}");
        assert_eq!(kept.shape()[axis], 1);

        let folded: Ndarr<i32, U3> = array.fold_axis(Keep(axis), 0, |a, x| a + x).unwrap();
        assert_eq!(folded, kept, "fold_axis axis {axis}");

        let dynamic: Ndarr<i32, Dyn> = array
            .clone()
            .into_dyn()
            .reduce(Keep(axis), |a, b| a + b)
            .unwrap();
        assert_eq!(dynamic, kept);
    }
    // A bare axis still removes it, with the rank checked statically.
    let dropped: Ndarr<i32, U2> = array.reduce(1, |a, b| a + b).unwrap();
    assert_eq!(dropped.shape(), &[2, 4]);
}

#[test]
fn kept_reductions_broadcast_back_against_their_input() {
    let array =
        Ndarr::<f64, U3>::new(&(0..24).map(f64::from).collect::<Vec<_>>(), [2, 3, 4]).unwrap();
    for axis in 0..3 {
        let centered = &array
            - &array.fold_axis(Keep(axis), 0.0, |a, x| a + x).unwrap() / array.shape()[axis] as f64;
        let residual = centered.reduce(axis, |a, b| a + b).unwrap();
        assert!(residual.iter_elems().all(|r| r.abs() < 1e-9), "axis {axis}");
    }
    // Views and strided layouts reduce in logical order too.
    let reversed = array.slice(s![.., ..;-1, ..]).unwrap();
    assert_eq!(
        reversed.reduce(Keep(1), f64::max).unwrap(),
        array.reduce(Keep(1), f64::max).unwrap()
    );
}

#[test]
fn keep_reports_bad_axes_and_keeps_empty_axes_with_their_seed() {
    let array = asymmetric();
    assert!(array.reduce(Keep(3), |a, b| a + b).is_err());
    assert!(array.fold_axis(Keep(3), 0, |a, x| a + x).is_err());

    let empty = Ndarr::<i32, U3>::new(&[], [2, 0, 4]).unwrap();
    assert!(empty.reduce(Keep(1), |a, b| a + b).is_err());
    let seeds = empty.fold_axis(Keep(1), 7, |a, x| a + x).unwrap();
    assert_eq!(seeds.shape(), &[2, 1, 4]);
    assert!(seeds.iter_elems().all(|&v| v == 7));
}
