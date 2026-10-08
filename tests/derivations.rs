//! Common selection verbs are short compositions of the existing algebra at
//! the same O(output) cost. Data-dependent selection can
//! never be a strided view; sorting, filtering, and membership belong to std.
use rapl::{s, Ndarr};
use typenum::U3;

#[test]
fn gather_is_keepdim_slices_plus_concatenate() {
    let a = Ndarr::from([[1, 2], [3, 4], [5, 6]]);
    let idx = [2isize, 0, 2];
    let rows: Vec<_> = idx
        .iter()
        .map(|&i| a.slice(s![i..i + 1, ..]).unwrap())
        .collect();
    let picked = Ndarr::concatenate(0, &rows).unwrap();
    assert_eq!(picked, Ndarr::from([[5, 6], [1, 2], [5, 6]]));
}

#[test]
fn compress_is_mask_positions_plus_gather() {
    let a = Ndarr::from([[1, 2], [3, 4], [5, 6]]);
    let mask = [true, false, true];
    let kept: Vec<_> = mask
        .iter()
        .enumerate()
        .filter(|&(_, &m)| m)
        .map(|(i, _)| a.slice(s![i as isize..i as isize + 1, ..]).unwrap())
        .collect();
    let compressed = Ndarr::concatenate(0, &kept).unwrap();
    assert_eq!(compressed, Ndarr::from([[1, 2], [5, 6]]));
}

#[test]
fn sort_axis_is_map_lanes_plus_std_sort() {
    let a = Ndarr::from([[3, 1, 2], [9, 7, 8]]);
    let sorted = a
        .map_lanes(1, |lane| {
            let dim = lane.dim().clone();
            let mut values: Vec<i32> = lane.iter_elems().cloned().collect();
            values.sort_unstable();
            Ndarr::from_vec_dim(values, dim).unwrap()
        })
        .unwrap();
    assert_eq!(sorted, Ndarr::from([[1, 2, 3], [7, 8, 9]]));
}

#[test]
fn grade_up_is_std_sort_of_indices() {
    let lane: Vec<i32> = vec![30, 10, 20];
    let mut grade: Vec<usize> = (0..lane.len()).collect();
    grade.sort_by_key(|&i| lane[i]);
    assert_eq!(grade, [1, 2, 0]);
}

#[test]
fn scatter_is_a_loop_over_indexed_assignment() {
    let mut a = Ndarr::from([0, 0, 0, 0]);
    for (&i, v) in [3usize, 0].iter().zip([9, 7]) {
        a[[i]] = v;
    }
    assert_eq!(a, Ndarr::from([7, 0, 0, 9]));
}

#[test]
fn cell_application_is_index_axis_plus_concatenate() {
    // Per-cell transpose over a rank-3 batch: the "rank operator" as a loop.
    let batch = Ndarr::<i32, U3>::new(&(0..12).collect::<Vec<_>>(), [3, 2, 2]).unwrap();
    let cells: Vec<Ndarr<i32, U3>> = (0..3)
        .map(|i| {
            let cell = batch.index_axis_view(0, i).unwrap();
            let transposed = cell.t_view();
            let reframed = transposed.insert_axis_view(0).unwrap();
            reframed.to_owned_array()
        })
        .collect();
    let out = Ndarr::concatenate(0, &cells).unwrap();
    assert_eq!(out.shape(), &[3, 2, 2]);
    assert_eq!(out.data(), &[0, 2, 1, 3, 4, 6, 5, 7, 8, 10, 9, 11]);
}
