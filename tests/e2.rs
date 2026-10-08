use rapl::{s, Dim, Dyn, Ndarr, NewAxis, SliceSpec};
use std::fmt::{Display, Formatter};
use std::ops::{Add, Mul};
use typenum::{U0, U1, U2, U3};

// No `Default` or `Debug` on purpose: pins the minimal element bounds of the APIs below.
#[derive(Clone, PartialEq)]
struct Lean(i32);

impl Add for Lean {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Lean(self.0 + rhs.0)
    }
}

impl Mul for Lean {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Lean(self.0 * rhs.0)
    }
}

impl Display for Lean {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

#[test]
fn structural_and_dyadic_apis_need_neither_default_nor_debug() {
    let mut vector: Ndarr<Lean, U1> = Ndarr::fill(Lean(2), [3]);
    vector[[1]] = Lean(5);
    assert!(vector.data() == [Lean(2), Lean(5), Lean(2)]);
    assert!(vector.roll(1, 0).data() == [Lean(2), Lean(2), Lean(5)]);
    assert!(vector.broadcast_view_to(&Dim::from([2, 3])).unwrap().len() == 6);
    assert_eq!(format!("{vector}"), "┌→────┐\n│2 5 2│\n└~────┘");

    let rhs = Ndarr::new(&[Lean(1), Lean(1), Lean(1)], [3]).unwrap();
    vector += &rhs;
    assert!(vector.data() == [Lean(3), Lean(6), Lean(3)]);

    let matrix = Ndarr::new(&[Lean(1), Lean(2), Lean(3), Lean(4)], [2, 2]).unwrap();
    let product = matrix.mat_mul(&matrix).unwrap();
    assert!(product.data() == [Lean(7), Lean(10), Lean(15), Lean(22)]);
}

#[test]
fn views_feed_reductions_and_float_maps_directly() {
    let a = Ndarr::from([[1, 2, 3], [4, 5, 6]]);
    // Views walk elements in logical order, so global reductions are std Iterator adapters.
    assert_eq!(a.t_view().iter_elems().sum::<i32>(), 21);
    assert_eq!(a.t_view().iter_elems().cloned().max(), Some(6));
    let row = Ndarr::from([3.0f64, -1.0, 2.0]);
    let view = row
        .broadcast_view_to(&Dim::from([2usize, 3]))
        .unwrap()
        .to_owned_array();
    assert_eq!(view.iter_elems().cloned().reduce(f64::max), Some(3.0));
    assert_eq!(view.iter_elems().cloned().reduce(f64::min), Some(-1.0));
}

#[test]
fn element_access_borrows_and_map_never_clones_input() {
    // `map` borrows: non-`Copy` elements map through `&String` with no clone.
    let words: Ndarr<String, U1> = Ndarr::new(&["a".to_string(), "bc".to_string()], [2]).unwrap();
    assert_eq!(words.map(|w| w.len()).data(), &[1, 2]);

    // `Index` / `IndexMut` work on views, not just owned arrays.
    let a = Ndarr::from([[1, 2, 3], [4, 5, 6]]);
    let t = a.t_view();
    assert_eq!(t[[2, 1]], 6);
    assert_eq!(t[[1, 0]], 2);

    let mut b = Ndarr::from([[1, 2], [3, 4]]);
    let mut col = b.slice_mut(s![.., 0]).unwrap();
    col[[1]] = 30;
    col[[0]] = 10;
    assert_eq!(b.data(), &[10, 2, 30, 4]);
}

#[test]
fn extent_one_axes_do_not_defeat_contiguity() {
    // [3] broadcast to [1, 3] has strides [0, 1] but is still buffer[0..3].
    let row = Ndarr::from([1, 2, 3]);
    let v = row.broadcast_view_to(&Dim::from([1usize, 3])).unwrap();
    assert!(v.is_standard_layout());
    assert_eq!(v.map(|x| x * 2).data(), &[2, 4, 6]);
    // A genuinely repeating broadcast stays non-contiguous.
    let w = row.broadcast_view_to(&Dim::from([2usize, 3])).unwrap();
    assert!(!w.is_standard_layout());
    assert_eq!(w.map(|x| x * 2).data(), &[2, 4, 6, 2, 4, 6]);
}

#[test]
fn approximate_comparison_is_symmetric_and_absolute() {
    let low = Ndarr::from([0.0]);
    let high = Ndarr::from([1.0]);
    assert!(!low.approx_epsilon(&high, 0.5));
    assert!(!high.approx_epsilon(&low, 0.5));
    assert!(low.approx_epsilon(&Ndarr::from([0.25]), 0.5));
}

#[test]
fn u0_and_dyn_have_distinct_meanings() {
    let scalar = Ndarr::<i32, U0>::new(&[7], Dim::<U0>::new(&[]).unwrap()).unwrap();
    assert_eq!(scalar.shape(), &[]);
    assert_eq!(scalar.scalar(), 7);
    assert!(Dim::<U0>::new(&[1]).is_err());

    let dynamic =
        Ndarr::<i32, Dyn>::new(&[1, 2, 3, 4, 5, 6], Dim::<Dyn>::new(&[2, 3]).unwrap()).unwrap();
    assert_eq!(dynamic.shape(), &[2, 3]);
}

#[test]
fn fixed_dynamic_transitions_move_the_element_buffer() {
    let fixed = Ndarr::<i32, U3>::new(&(0..24).collect::<Vec<_>>(), [2, 3, 4]).unwrap();
    let pointer = fixed.data().as_ptr();
    let dynamic = fixed.into_dyn();
    assert_eq!(dynamic.data().as_ptr(), pointer);
    let fixed = dynamic.into_ranked::<U3>().unwrap();
    assert_eq!(fixed.data().as_ptr(), pointer);
    assert!(fixed.into_dyn().into_ranked::<U2>().is_err());
}

#[test]
fn insert_axis_view_infers_fixed_and_dynamic_output_ranks() {
    let fixed = Ndarr::<i32, U2>::new(&(0..6).collect::<Vec<_>>(), [2, 3]).unwrap();
    let inserted: rapl::NdView<'_, i32, U3> = fixed.insert_axis_view(1).unwrap();
    assert_eq!(inserted.shape(), &[2, 1, 3]);
    // An extent-1 axis carries no layout information, so its stride is zero.
    assert_eq!(inserted.strides(), &[3, 0, 1]);
    assert_eq!(
        inserted.iter_elems().cloned().collect::<Vec<_>>(),
        fixed.data()
    );

    let restored: rapl::NdView<'_, i32, U2> = inserted.index_axis_view(1, 0).unwrap();
    assert_eq!(restored.shape(), fixed.shape());
    assert_eq!(restored.strides(), fixed.strides());
    assert_eq!(restored.offset(), fixed.offset());

    let dynamic = fixed.clone().into_dyn();
    let inserted: rapl::NdView<'_, i32, Dyn> = dynamic.insert_axis_view(0).unwrap();
    assert_eq!(inserted.shape(), &[1, 2, 3]);
    assert_eq!(
        inserted.iter_elems().cloned().collect::<Vec<_>>(),
        dynamic.data()
    );
}

#[test]
fn insert_axis_view_supports_scalars_and_rejects_invalid_axes() {
    let scalar = Ndarr::<i32, U0>::new(&[7], []).unwrap();
    let vector: rapl::NdView<'_, i32, U1> = scalar.insert_axis_view(0).unwrap();
    assert_eq!(vector.shape(), &[1]);
    assert_eq!(vector[[0]], 7);

    assert!(scalar.insert_axis_view(1).is_err());
}

#[test]
fn slicing_supports_ranges_negative_bounds_and_steps() {
    let array = Ndarr::<i32, U2>::new(&(0..20).collect::<Vec<_>>(), [4, 5]).unwrap();

    let middle = array.slice(s![1..4, 1..5]).unwrap();
    assert_eq!(middle.shape(), &[3, 4]);
    assert_eq!(
        middle.iter_elems().cloned().collect::<Vec<_>>(),
        &[6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19]
    );

    let reversed = array
        .slice(s![
            -1..;-2,
            ..;-2,
        ])
        .unwrap();
    assert_eq!(reversed.shape(), &[2, 3]);
    assert_eq!(
        reversed.iter_elems().cloned().collect::<Vec<_>>(),
        &[19, 17, 15, 9, 7, 5]
    );

    let clamped = array.slice(s![-100..100, 100..,]).unwrap();
    assert_eq!(clamped.shape(), &[4, 0]);
    assert!(clamped.iter_elems().next().is_none());
}

#[test]
fn slicing_validates_specs_and_step() {
    let array = Ndarr::<i32, U2>::new(&(0..6).collect::<Vec<_>>(), [2, 3]).unwrap();
    assert!(array.slice(s![..]).is_err());
    assert!(array.slice(s![.., ..;0]).is_err());
}

#[test]
fn slicing_composes_with_transpose_and_broadcast_views() {
    let array = Ndarr::<i32, U2>::new(&(0..12).collect::<Vec<_>>(), [3, 4]).unwrap();
    let transposed = array.t_view();
    let slice = transposed.slice(s![1..4, ..;-1]).unwrap();
    assert_eq!(slice.shape(), &[3, 3]);
    assert_eq!(
        slice.iter_elems().cloned().collect::<Vec<_>>(),
        &[9, 5, 1, 10, 6, 2, 11, 7, 3]
    );

    let row = Ndarr::from([1, 2, 3]);
    let broadcast = row.broadcast_view_to(&Dim::from([2, 3])).unwrap();
    let slice = broadcast.slice(s![.., ..;-1]).unwrap();
    assert_eq!(
        slice.iter_elems().cloned().collect::<Vec<_>>(),
        &[3, 2, 1, 3, 2, 1]
    );
}

/// Adapters lend the view's own lifetime, not the temporary's, so chains are one expression.
#[test]
fn view_adapters_chain_and_outlive_their_temporaries() {
    fn lower_right<'a>(v: rapl::NdView<'a, i32, U2>) -> rapl::NdView<'a, i32, U2> {
        v.slice(s![1.., 1..]).unwrap().t_view()
    }

    let array = Ndarr::<i32, U2>::new(&(0..12).collect::<Vec<_>>(), [3, 4]).unwrap();
    let chained = array
        .view()
        .slice(s![1.., ..])
        .unwrap()
        .view()
        .permute_axes(&[1, 0])
        .unwrap()
        .slice(s![..2, ..])
        .unwrap();
    assert_eq!(chained.shape(), &[2, 2]);
    assert_eq!(
        chained.iter_elems().cloned().collect::<Vec<_>>(),
        &[4, 8, 5, 9]
    );

    let corner = lower_right(array.view());
    assert_eq!(corner.shape(), &[3, 2]);
    assert_eq!(corner[[2, 1]], 11);

    // Owned and mutable storage still lend a borrow of `self`.
    let mut owned = array.clone();
    let from_owned: rapl::NdView<'_, i32, U1> = owned.t_view().index_axis_view(0, 0).unwrap();
    assert_eq!(
        from_owned.iter_elems().cloned().collect::<Vec<_>>(),
        &[0, 4, 8]
    );
    let mutable = owned.view_mut();
    let from_mut: rapl::NdView<'_, i32, U1> = mutable.index_axis_view(1, 2).unwrap();
    assert_eq!(
        from_mut.iter_elems().cloned().collect::<Vec<_>>(),
        &[2, 6, 10]
    );
}

#[test]
fn mutable_slice_writes_through_negative_strides() {
    let mut array = Ndarr::<i32, U2>::new(&(0..12).collect::<Vec<_>>(), [3, 4]).unwrap();
    {
        let mut view = array.slice_mut(s![1..3, ..;-2]).unwrap();
        assert_eq!(view.shape(), &[2, 2]);
        view[[0, 0]] = 100;
        view[[1, 1]] = 200;
    }
    assert_eq!(array.data(), &[0, 1, 2, 3, 4, 5, 6, 100, 8, 200, 10, 11]);
}

#[test]
fn integer_selectors_remove_axes_and_derive_index_axis_view() {
    let array = Ndarr::<i32, U3>::new(&(0..24).collect::<Vec<_>>(), [2, 3, 4]).unwrap();

    // `s![i, ..]` is the grammar form of `index_axis_view(0, i)`.
    let typed: rapl::NdView<'_, i32, U2> = array.slice(s![1, .., ..]).unwrap();
    let direct = array.index_axis_view(0, 1).unwrap();
    assert_eq!(typed.shape(), direct.shape());
    assert_eq!(typed.offset(), direct.offset());
    assert_eq!(typed.strides(), direct.strides());

    // Negative indexes wrap once from the end.
    let last: rapl::NdView<'_, i32, U2> = array.slice(s![.., -1, ..]).unwrap();
    assert_eq!(last, array.index_axis_view(1, 2).unwrap());
    assert!(array.slice(s![.., -4, ..]).is_err());

    // All-integer selection reaches rank zero.
    let scalar: rapl::NdView<'_, i32, U0> = array.slice(s![1, 2, 3]).unwrap();
    assert_eq!(scalar[[]], 23);

    let mixed: rapl::NdView<'_, i32, U1> = array.slice(s![1, ..;-1, 2]).unwrap();
    assert_eq!(
        mixed.iter_elems().cloned().collect::<Vec<_>>(),
        vec![22, 18, 14]
    );
}

#[test]
fn new_axis_selector_derives_insert_axis_view() {
    let array = Ndarr::<i32, U2>::new(&(0..6).collect::<Vec<_>>(), [2, 3]).unwrap();
    let typed: rapl::NdView<'_, i32, U3> = array.slice(s![.., NewAxis, ..]).unwrap();
    let direct = array.insert_axis_view(1).unwrap();
    assert_eq!(typed.shape(), direct.shape());
    assert_eq!(typed, direct);

    // `NewAxis` consumes no input axis, so it can also lead or trail.
    let framed: rapl::NdView<'_, i32, rapl::U4> =
        array.slice(s![NewAxis, .., .., NewAxis]).unwrap();
    assert_eq!(framed.shape(), &[1, 2, 3, 1]);
    assert_eq!(
        framed.iter_elems().cloned().collect::<Vec<_>>(),
        array.iter_elems().cloned().collect::<Vec<_>>()
    );
}

#[test]
fn dynamic_arrays_slice_through_both_impl_families() {
    let array = Ndarr::<i32, Dyn>::new(
        &(0..24).collect::<Vec<_>>(),
        Dim::<Dyn>::new(&[2, 3, 4]).unwrap(),
    )
    .unwrap();
    // A typed tuple recovers static rank even from a `Dyn` input.
    let slice: rapl::NdView<'_, i32, U3> = array.slice(s![.., 1..3, ..;-1]).unwrap();
    assert_eq!(slice.shape(), &[2, 2, 4]);
    // Runtime-built specs are the `Dyn` escape hatch.
    let specs: Vec<SliceSpec> = vec![(..).into(), (1..3).into(), SliceSpec::Index(-1)];
    let slice: rapl::NdView<'_, i32, Dyn> = array.slice(specs.as_slice()).unwrap();
    assert_eq!(slice.shape(), &[2, 2]);
}

#[test]
fn reshape_and_reduce_accept_views() {
    let array = Ndarr::<i32, U2>::new(&(0..12).collect::<Vec<_>>(), [3, 4]).unwrap();

    // Non-contiguous view: reshape never copies, so materialize first.
    let transposed = array.t_view();
    assert!(transposed.view().reshape([2, 6]).is_err());
    let reshaped: Ndarr<i32, U2> = transposed.to_owned_array().reshape([2, 6]).unwrap();
    assert_eq!(reshaped, array.t().reshape([2, 6]).unwrap());

    // Contiguous view: reshape stays an O(1) view.
    let view = array.view();
    let flat: rapl::NdView<'_, i32, U1> = view.reshape([12]).unwrap();
    assert_eq!(flat.iter_elems().cloned().collect::<Vec<_>>(), array.data());

    let reduced = transposed.reduce(1, |x, y| x + y).unwrap();
    assert_eq!(reduced, array.t().reduce(1, |x, y| x + y).unwrap());
}

#[test]
fn zip_with_and_operators_accept_views() {
    let matrix = Ndarr::from([[1, 2], [3, 4]]);
    let row = Ndarr::from([10, 20]);

    let summed = matrix.t_view().zip_with(&row.view(), |x, y| x + y).unwrap();
    assert_eq!(summed, Ndarr::from([[11, 23], [12, 24]]));

    assert_eq!(
        matrix.t_view() + row.view(),
        Ndarr::from([[11, 23], [12, 24]])
    );
    assert_eq!(&matrix.t_view() - &row.view(), matrix.t() - &row);
    assert_eq!(matrix.view() * 2, Ndarr::from([[2, 4], [6, 8]]));
    assert_eq!(2 * matrix.view(), Ndarr::from([[2, 4], [6, 8]]));
    assert_eq!(-matrix.view(), Ndarr::from([[-1, -2], [-3, -4]]));
}

#[test]
fn assign_operators_accept_view_operands() {
    let matrix = Ndarr::from([[1, 2], [3, 4]]);
    let mut accumulator = Ndarr::from([[10, 20], [30, 40]]);
    accumulator += &matrix.t_view();
    assert_eq!(accumulator, Ndarr::from([[11, 23], [32, 44]]));

    let mut buffer = Ndarr::from([[1, 2], [3, 4]]);
    let mut view = buffer.view_mut();
    view += 100;
    assert_eq!(buffer, Ndarr::from([[101, 102], [103, 104]]));
}

/// Without the `Scalar` bound, a rank-k literal would also match the rank-(k-1) impl and inference would fail.
#[test]
fn nested_literals_infer_one_rank_each() {
    let matrix = Ndarr::from([[1, 2], [3, 4]]);
    assert_eq!(matrix.shape(), &[2, 2]);
    let cube = Ndarr::from([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]);
    assert_eq!(cube.shape(), &[2, 2, 2]);
    let hyper = Ndarr::from([[[[1, 2]]], [[[3, 4]]]]);
    assert_eq!(hyper.shape(), &[2, 1, 1, 2]);
}

/// `Vec`/`Range` sources have no nesting ambiguity: non-`Scalar` elements convert directly.
#[test]
fn vec_sources_accept_non_scalar_elements() {
    let words = Ndarr::from(vec!["a".to_string(), "bc".to_string()]);
    assert_eq!(words.shape(), &[2]);
    assert_eq!(words.map(|w| w.len()).data(), &[1, 2]);
    let counted = Ndarr::from(0..4);
    assert_eq!(counted.data(), &[0, 1, 2, 3]);
}

/// Combination outputs never need `Clone`: produced values move straight into the output buffer.
#[test]
fn combination_outputs_need_no_clone() {
    #[derive(PartialEq, Debug)]
    struct Acc(i32);

    let a = Ndarr::from([1, 2]);
    let b = Ndarr::from([10, 20]);
    let summed = a.zip_with(&b, |x, y| Acc(x + y)).unwrap();
    assert_eq!(summed.data(), &[Acc(11), Acc(22)]);

    let outer = a.outer_product(&b, |x, y| Acc(x * y)).unwrap();
    assert_eq!(outer.shape(), &[2, 2]);
    assert_eq!(outer.data(), &[Acc(10), Acc(20), Acc(20), Acc(40)]);

    let inner = a
        .inner_product(&b, |x, y| Acc(x * y), |p, q| Acc(p.0 + q.0))
        .unwrap();
    assert_eq!(inner.data(), &[Acc(50)]);
}

#[test]
fn mixed_rank_broadcasts_produce_dyn() {
    let fixed = Ndarr::from([1, 2, 3]);
    let dynamic =
        Ndarr::<i32, Dyn>::new(&[10, 20, 30, 40, 50, 60], Dim::<Dyn>::new(&[2, 3]).unwrap())
            .unwrap();
    let left: Ndarr<i32, Dyn> = &fixed + &dynamic;
    let right: Ndarr<i32, Dyn> = &dynamic + &fixed;
    assert_eq!(left.shape(), &[2, 3]);
    assert_eq!(left.data(), right.data());
}
