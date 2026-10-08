use rapl::{s, Dim, Dyn, NdView, Ndarr, Win, U0, U1, U2, U3};

fn asymmetric() -> Ndarr<i32, U3> {
    Ndarr::new(&(0..24).collect::<Vec<_>>(), [2, 3, 4]).unwrap()
}

/// Reference: every cell visited by direct coordinate lookup.
fn cells_by_lookup(array: &Ndarr<i32, U3>, k: usize) -> Vec<Vec<i32>> {
    let shape = array.shape();
    let (frame, cell) = shape.split_at(3 - k);
    let count = |extents: &[usize]| extents.iter().product::<usize>();
    let unflatten = |mut flat: usize, extents: &[usize]| {
        let mut index = vec![0; extents.len()];
        for axis in (0..extents.len()).rev() {
            index[axis] = flat % extents[axis];
            flat /= extents[axis];
        }
        index
    };
    (0..count(frame))
        .map(|f| {
            (0..count(cell))
                .map(|c| {
                    let mut index = unflatten(f, frame);
                    index.extend(unflatten(c, cell));
                    array[index.as_slice()]
                })
                .collect()
        })
        .collect()
}

#[test]
fn cells_are_trailing_axis_views_in_logical_order() {
    let array = asymmetric();
    for k in 0..=3 {
        let mut visited = Vec::new();
        let out = array
            .map_cells(k, |cell| {
                assert_eq!(cell.shape(), &array.shape()[3 - k..]);
                visited.push(cell.iter_elems().copied().collect::<Vec<_>>());
                visited.len() - 1
            })
            .unwrap();
        assert_eq!(out.shape(), &array.shape()[..3 - k]);
        assert_eq!(out.into_data(), (0..visited.len()).collect::<Vec<_>>());
        assert_eq!(visited, cells_by_lookup(&array, k), "k = {k}");
    }
}

#[test]
fn typenum_counts_keep_fixed_ranks() {
    let array = asymmetric();
    let sums: Ndarr<i32, U1> = array
        .map_cells(U2::new(), |cell: NdView<'_, i32, U2>| {
            cell.iter_elems().sum::<i32>()
        })
        .unwrap();
    assert_eq!(sums, Ndarr::from([66, 210]));
    let whole: Ndarr<i32, U0> = array
        .map_cells(U3::new(), |cell| cell.iter_elems().sum::<i32>())
        .unwrap();
    assert_eq!(whole.scalar(), 276);
    let elements: Ndarr<i32, U3> = array.map_cells(U0::new(), |cell| cell[[]]).unwrap();
    assert_eq!(elements, array);
}

#[test]
fn dynamic_ranks_and_runtime_counts_select_dyn() {
    let array = asymmetric();
    let dynamic = array.clone().into_dyn();
    let typed_cells: Ndarr<i32, Dyn> = dynamic
        .map_cells(U2::new(), |cell: NdView<'_, i32, U2>| cell[[2, 3]])
        .unwrap();
    assert_eq!(typed_cells.shape(), &[2]);
    assert_eq!(typed_cells.into_data(), vec![11, 23]);
    let runtime: Ndarr<i32, Dyn> = array
        .map_cells(2usize, |cell: NdView<'_, i32, Dyn>| cell[[2, 3]])
        .unwrap();
    assert_eq!(runtime.shape(), &[2]);
    assert!(dynamic.map_cells(U3::new(), |_| ()).is_ok());
}

#[test]
fn counts_beyond_a_runtime_rank_fail_before_callbacks() {
    let dynamic = asymmetric().into_dyn();
    let mut calls = 0;
    assert!(dynamic.map_cells(4usize, |_| calls += 1).is_err());
    assert!(asymmetric().map_cells(4usize, |_| calls += 1).is_err());
    let rank_two = Ndarr::from([[1, 2], [3, 4]]).into_dyn();
    assert!(rank_two.map_cells(U3::new(), |_| calls += 1).is_err());
    assert_eq!(calls, 0);
}

#[test]
fn cells_follow_strided_overlapping_and_reversed_views() {
    let array = Ndarr::new(&(0..20).collect::<Vec<_>>(), [4, 5]).unwrap();
    // Overlapping 2x3 windows at stride 2 along the columns: [row, col, kh, kw].
    let windows = array
        .slice(s![Win(2), Win(3)])
        .unwrap()
        .view()
        .permute_axes(&[0, 2, 1, 3])
        .unwrap()
        .slice(s![.., ..;2, .., ..])
        .unwrap();
    let corners = windows
        .map_cells(U2::new(), |window| (window[[0, 0]], window[[1, 2]]))
        .unwrap();
    assert_eq!(corners.shape(), &[3, 2]);
    for row in 0..3 {
        for col in 0..2 {
            let (top_left, bottom_right) = corners[[row, col]];
            assert_eq!(top_left, array[[row, 2 * col]]);
            assert_eq!(bottom_right, array[[row + 1, 2 * col + 2]]);
        }
    }

    let reversed = array.slice(s![..;-1, ..]).unwrap();
    let firsts = reversed.map_cells(U1::new(), |row| row[[0]]).unwrap();
    assert_eq!(firsts, Ndarr::from([15, 10, 5, 0]));

    let broadcast = Ndarr::from([1, 2, 3])
        .broadcast_view_to(&Dim::<U2>::new(&[2, 3]).unwrap())
        .unwrap()
        .map_cells(U1::new(), |row| row.iter_elems().sum::<i32>())
        .unwrap();
    assert_eq!(broadcast, Ndarr::from([6, 6]));
}

#[test]
fn empty_cells_reach_the_callback_and_empty_frames_do_not() {
    let empty_cells = Ndarr::<i32, U2>::new(&[], [3, 0]).unwrap();
    let mut calls = 0;
    let lengths = empty_cells
        .map_cells(U1::new(), |cell| {
            calls += 1;
            cell.len()
        })
        .unwrap();
    assert_eq!((calls, lengths), (3, Ndarr::from([0, 0, 0])));

    let empty_frame = Ndarr::<i32, U3>::new(&[], [0, 2, 2]).unwrap();
    let out = empty_frame
        .map_cells(U2::new(), |_| -> i32 {
            panic!("an empty frame has no cells")
        })
        .unwrap();
    assert_eq!(out.shape(), &[0]);
}

#[test]
fn lanes_are_the_one_axis_cells_of_a_permutation() {
    let array = asymmetric();
    for axis in 0..3 {
        let mut order: Vec<usize> = (0..3).filter(|&a| a != axis).collect();
        order.push(axis);
        let from_lanes: Vec<Vec<i32>> = array
            .lanes(axis)
            .unwrap()
            .map(|lane| lane.iter_elems().copied().collect())
            .collect();
        let from_cells = array
            .view()
            .permute_axes(&order)
            .unwrap()
            .map_cells(U1::new(), |lane| {
                lane.iter_elems().copied().collect::<Vec<_>>()
            })
            .unwrap()
            .into_data();
        assert_eq!(from_lanes, from_cells, "axis = {axis}");
    }
}
