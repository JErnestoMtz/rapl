//! `s!` accepts every primitive integer type for indices, bounds, and steps,
//! and `range;step` for stepped selections, all lowered through one `Slice`.

use rapl::{s, Dyn, Ndarr, Slice, SliceSpec, U1, U2};

fn grid() -> Ndarr<i32, U2> {
    Ndarr::new(&(0..20).collect::<Vec<_>>(), [4, 5]).unwrap()
}

fn row(view: rapl::NdView<'_, i32, U1>) -> Vec<i32> {
    view.iter_elems().copied().collect()
}

#[test]
fn every_integer_type_selects_the_same_view() {
    let a = grid();
    let expected = a.slice(s![1isize, 1isize..4isize]).unwrap();
    assert_eq!(a.slice(s![1usize, 1usize..4usize]).unwrap(), expected);
    assert_eq!(a.slice(s![1u8, 1u8..4u8]).unwrap(), expected);
    assert_eq!(a.slice(s![1i64, 1i64..4i64]).unwrap(), expected);
    assert_eq!(a.slice(s![1u32, 1u16..4u16]).unwrap(), expected);
    // Literals still infer, including negative ones.
    assert_eq!(a.slice(s![1, 1..4]).unwrap(), expected);
    assert_eq!(row(a.slice(s![-1, ..-3]).unwrap()), vec![15, 16]);
    // `usize` extents need no casts.
    let (start, len) = (2usize, a.shape()[1]);
    assert_eq!(row(a.slice(s![0, start..len]).unwrap()), vec![2, 3, 4]);
}

#[test]
// Descending and empty bounds are deliberate here; clippy reads literal
// `start > end` ranges as mistakes.
#[allow(clippy::reversed_empty_ranges)]
fn stepped_ranges_match_numpy_selections() {
    let a = grid();
    let first_row = |selection: Slice| row(a.slice(s![0, selection]).unwrap());
    let cases: [(Slice, Vec<i32>); 9] = [
        (Slice::stepped(.., 2), vec![0, 2, 4]),
        (Slice::stepped(1.., 2), vec![1, 3]),
        (Slice::stepped(..4, 3), vec![0, 3]),
        (Slice::stepped(1..5, 2), vec![1, 3]),
        (Slice::stepped(.., -1), vec![4, 3, 2, 1, 0]),
        (Slice::stepped(-1.., -2), vec![4, 2, 0]),
        (Slice::stepped(3..0, -1), vec![3, 2, 1]),
        (Slice::stepped(3..1, 1), vec![]),
        (Slice::stepped(1..3, -1), vec![]),
    ];
    for (selection, expected) in cases {
        assert_eq!(first_row(selection), expected, "{selection:?}");
    }
    // The macro spelling is the same `Slice`.
    assert_eq!(row(a.slice(s![0, ..;2]).unwrap()), vec![0, 2, 4]);
    assert_eq!(row(a.slice(s![0, 1..5;2]).unwrap()), vec![1, 3]);
    assert_eq!(row(a.slice(s![0, -1..;-2]).unwrap()), vec![4, 2, 0]);
    let stride = 2usize;
    assert_eq!(
        a.slice(s![..;stride, 1..;stride]).unwrap(),
        Ndarr::from([[1, 3], [11, 13]])
    );
    assert!(a.slice(s![.., ..;0]).is_err());
}

#[test]
fn values_beyond_isize_saturate_to_the_same_selection() {
    let a = grid();
    assert_eq!(row(a.slice(s![0, 1..u64::MAX]).unwrap()), vec![1, 2, 3, 4]);
    assert_eq!(row(a.slice(s![0, ..;usize::MAX]).unwrap()), vec![0]);
    assert_eq!(row(a.slice(s![0, ..;i64::MIN]).unwrap()), vec![4]);
    assert!(a.slice(s![usize::MAX, ..]).is_err());
}

#[test]
fn stepped_selections_are_mutable_and_runtime_specs_share_the_grammar() {
    let mut a = grid();
    let mut corners = a.slice_mut(s![..;3, ..;4]).unwrap();
    corners += 100;
    assert_eq!(a[[0, 0]] + a[[0, 4]] + a[[3, 0]] + a[[3, 4]], 438);

    let dynamic: Ndarr<i32, Dyn> = grid().into_dyn();
    let specs = [SliceSpec::from(2usize), Slice::stepped(.., -2).into()];
    assert_eq!(
        dynamic
            .slice(&specs[..])
            .unwrap()
            .iter_elems()
            .copied()
            .collect::<Vec<_>>(),
        vec![14, 12, 10]
    );
}
