//! Complex arithmetic and display, checked against golden values and mathematical identities.
#![cfg(feature = "complex")]

use rapl::{Dim, Imag, Ndarr, StaticRank, C, U0, U1, U2, U3, U4};

const N_RANDOM: usize = 48;
const EPS_64: f64 = 1e-15;
const EPS_32: f32 = 1e-6;

/// Small deterministic xorshift64* PRNG.
struct Prng(u64);

impl Prng {
    fn new() -> Self {
        Prng(0xABC0_FFEE_0142)
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn f64_in(&mut self, lo: f64, hi: f64) -> f64 {
        let u = (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
        lo + u * (hi - lo)
    }
    fn usize_in(&mut self, lo: usize, hi_inclusive: usize) -> usize {
        lo + (self.next_u64() as usize) % (hi_inclusive - lo + 1)
    }
}

fn new_arr<T: Clone, R: StaticRank>(data: &[T], shape: &[usize]) -> Ndarr<T, R> {
    Ndarr::new(data, Dim::<R>::new(shape).unwrap()).unwrap()
}

/// NaN == NaN; relative tolerance with an absolute floor at magnitude 1.
fn same_f64(x: f64, y: f64, eps: f64) -> bool {
    x == y || (x.is_nan() && y.is_nan()) || ((x - y).abs() <= eps * x.abs().max(1.0))
}

fn same_f32(x: f32, y: f32, eps: f32) -> bool {
    x == y || (x.is_nan() && y.is_nan()) || ((x - y).abs() <= eps * x.abs().max(1.0))
}

fn approx_c64(a: C<f64>, b: C<f64>, eps: f64) -> bool {
    same_f64(a.0, b.0, eps) && same_f64(a.1, b.1, eps)
}

fn approx_c32(a: C<f32>, b: C<f32>, eps: f32) -> bool {
    same_f32(a.0, b.0, eps) && same_f32(a.1, b.1, eps)
}

fn assert_c64_slices(op: &str, got: &[C<f64>], expected: &[C<f64>]) {
    assert_eq!(got.len(), expected.len(), "{op}: length mismatch");
    for (i, (a, b)) in got.iter().zip(expected.iter()).enumerate() {
        assert!(
            approx_c64(*a, *b, EPS_64),
            "{op}: mismatch at {i}: got={a:?} expected={b:?}"
        );
    }
}

fn assert_c32_slices(op: &str, got: &[C<f32>], expected: &[C<f32>]) {
    assert_eq!(got.len(), expected.len(), "{op}: length mismatch");
    for (i, (a, b)) in got.iter().zip(expected.iter()).enumerate() {
        assert!(
            approx_c32(*a, *b, EPS_32),
            "{op}: mismatch at {i}: got={a:?} expected={b:?}"
        );
    }
}

fn assert_f64_slices(op: &str, got: &[f64], expected: &[f64]) {
    assert_eq!(got.len(), expected.len(), "{op}: length mismatch");
    for (i, (a, b)) in got.iter().zip(expected.iter()).enumerate() {
        assert!(
            same_f64(*a, *b, EPS_64),
            "{op}: mismatch at {i}: got={a:?} expected={b:?}"
        );
    }
}

/// The eight quadrant / axis probes plus branch-cut points on the negative real axis and zero.
fn probe_points() -> Vec<(f64, f64)> {
    vec![
        (1.0, 1.0),
        (-1.0, 1.0),
        (-1.0, -1.0),
        (1.0, -1.0),
        (3.5, 0.25),
        (-3.5, 0.25),
        (-3.5, -0.25),
        (3.5, -0.25),
        (0.0, 1.0),
        (0.0, -1.0),
        (2.0, 0.0),
        (-1.0, 0.0),
        (-2.5, 0.0),
        (-2.5, -0.0),
        (0.0, 0.0),
        (-0.0, 0.0),
    ]
}

fn probe_c64() -> Vec<C<f64>> {
    probe_points().into_iter().map(|(r, i)| C(r, i)).collect()
}

// f64 reference complex arithmetic from std functions only: the reference oracle.

type P = (f64, f64);

fn radd(a: P, b: P) -> P {
    (a.0 + b.0, a.1 + b.1)
}
fn rsub(a: P, b: P) -> P {
    (a.0 - b.0, a.1 - b.1)
}
fn rmul(a: P, b: P) -> P {
    (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
}
fn rdiv(a: P, b: P) -> P {
    let den = b.0 * b.0 + b.1 * b.1;
    ((a.0 * b.0 + a.1 * b.1) / den, (a.1 * b.0 - a.0 * b.1) / den)
}
fn rabs(z: P) -> f64 {
    z.0.hypot(z.1)
}
fn rarg(z: P) -> f64 {
    z.1.atan2(z.0)
}
/// Euler: e^{x+iy} = e^x (cos y + i sin y)
fn rexp(z: P) -> P {
    (z.0.exp() * z.1.cos(), z.0.exp() * z.1.sin())
}
fn rln(z: P) -> P {
    (rabs(z).ln(), rarg(z))
}
fn rsqrt(z: P) -> P {
    let (r, arg) = (rabs(z), rarg(z));
    rmul((r.sqrt(), 0.0), rexp((0.0, arg / 2.0)))
}
/// sin z = (e^{iz} - e^{-iz}) / 2i
fn rsin(z: P) -> P {
    rdiv(rsub(rexp((-z.1, z.0)), rexp((z.1, -z.0))), (0.0, 2.0))
}
/// cos z = (e^{iz} + e^{-iz}) / 2
fn rcos(z: P) -> P {
    rdiv(radd(rexp((-z.1, z.0)), rexp((z.1, -z.0))), (2.0, 0.0))
}
fn rtan(z: P) -> P {
    rdiv(rsin(z), rcos(z))
}
fn rsinh(z: P) -> P {
    rdiv(rsub(rexp(z), rexp((-z.0, -z.1))), (2.0, 0.0))
}
fn rcosh(z: P) -> P {
    rdiv(radd(rexp(z), rexp((-z.0, -z.1))), (2.0, 0.0))
}
fn rtanh(z: P) -> P {
    rdiv(rsinh(z), rcosh(z))
}
/// z^n = r^n e^{i n theta}
fn rpowf(z: P, n: f64) -> P {
    let (r, arg) = (rabs(z), rarg(z));
    let pow_r = r.powf(n);
    let pow_c = rexp((0.0, arg * n));
    (pow_r * pow_c.0, pow_r * pow_c.1)
}
/// z^w = e^{(ln r + i theta) w}
fn rpowc(z: P, c: P) -> P {
    let (r, arg) = (rabs(z), rarg(z));
    let p1 = (r.ln() * c.0, r.ln() * c.1);
    let p2 = rmul((0.0, arg), c);
    rexp(radd(p1, p2))
}

fn cp(z: P) -> C<f64> {
    C(z.0, z.1)
}
fn pc(z: &C<f64>) -> P {
    (z.0, z.1)
}

fn assert_display_golden<T, R>(case: &str, data: &[T], shape: &[usize], expected: &str)
where
    T: Clone + std::fmt::Debug + std::fmt::Display,
    R: StaticRank,
{
    let n = new_arr::<T, R>(data, shape);
    assert_eq!(
        format!("{}", n),
        expected,
        "Display mismatch [{case}] shape={shape:?}"
    );
}

#[test]
fn display_rank0() {
    assert_display_golden::<i32, U0>("rank0_pos", &[7], &[], "7");
    assert_display_golden::<i32, U0>("rank0_neg", &[-1234], &[], "-1234");
    assert_display_golden::<f64, U0>("rank0_float", &[-0.5], &[], "-0.5");
}

#[test]
fn display_rank1() {
    assert_display_golden::<i32, U1>("single", &[5], &[1], "┌→┐\n│5│\n└~┘");
    assert_display_golden::<i32, U1>("uniform", &[1, 2, 3], &[3], "┌→────┐\n│1 2 3│\n└~────┘");
    assert_display_golden::<i32, U1>(
        "mixed_width",
        &[1, -22, 333, -4444],
        &[4],
        "┌→──────────────┐\n│1 -22 333 -4444│\n└~──────────────┘",
    );
    // Under the 1000-element threshold nothing is elided.
    let s = format!(
        "{}",
        new_arr::<i32, U1>(&(0..1000).collect::<Vec<_>>(), &[1000])
    );
    assert!(!s.contains('·'), "{s}");
    // Past it, each end keeps three items around one ellipsis slot.
    let s = format!(
        "{}",
        new_arr::<i32, U1>(&(0..1001).collect::<Vec<_>>(), &[1001])
    );
    assert_eq!(s.lines().nth(1), Some("│0 1 2 · 998 999 1000│"), "{s}");
}

#[test]
fn display_rank2_asymmetric() {
    let d6: Vec<i32> = (0..6).collect();
    assert_display_golden::<i32, U2>("2x3", &d6, &[2, 3], "┌→────┐\n↓0 1 2│\n│3 4 5│\n└~────┘");
    let d5: Vec<i32> = vec![1, -22, 333, -4444, 55555];
    assert_display_golden::<i32, U2>(
        "5x1",
        &d5,
        &[5, 1],
        "┌→────┐\n↓    1│\n│  -22│\n│  333│\n│-4444│\n│55555│\n└~────┘",
    );
    assert_display_golden::<i32, U2>("1x1", &[42], &[1, 1], "┌→─┐\n↓42│\n└~─┘");
    // 40x40 elides both axes: 3 + ellipsis + 3 rows and columns.
    let a = new_arr::<i32, U2>(&(0..1600).collect::<Vec<_>>(), &[40, 40]);
    let s = format!("{a}");
    let lines: Vec<&str> = s.lines().collect();
    assert_eq!(lines.len(), 2 + 7, "{s}");
    assert_eq!(lines[1], "↓   0    1    2 ·   37   38   39│", "{s}");
    assert_eq!(lines[4], "│   ·    ·    · ·    ·    ·    ·│", "{s}");
}

#[test]
fn display_rank3_and_rank4_spacing() {
    assert_display_golden::<i32, U3>("1x1x1", &[11], &[1, 1, 1], "┌┌→─┐\n↓↓11│\n└└~─┘");
    // One blank line per leading axis that advanced between 2-D slices.
    assert_display_golden::<i32, U4>(
        "2x2x1x2",
        &(0..8).collect::<Vec<_>>(),
        &[2, 2, 1, 2],
        "┌┌┌→──┐\n↓↓↓0 1│\n│││   │\n│││2 3│\n│││   │\n│││   │\n│││4 5│\n│││   │\n│││6 7│\n└└└~──┘",
    );
}

#[test]
fn display_empty_arrays() {
    // ⊖ replaces the marker of each empty axis.
    assert_display_golden::<i32, U1>("0", &[], &[0], "┌⊖┐\n└~┘");
    assert_display_golden::<i32, U2>("0x3", &[], &[0, 3], "┌→┐\n⊖ │\n└~┘");
    assert_display_golden::<i32, U2>("3x0", &[], &[3, 0], "┌⊖┐\n↓ │\n│ │\n│ │\n└~┘");
    assert_display_golden::<i32, U3>("2x0x3", &[], &[2, 0, 3], "┌┌→┐\n↓⊖ │\n└└~┘");
}

#[test]
fn display_aligns_decimals_and_text() {
    assert_display_golden::<f64, U2>(
        "floats",
        &[1.0, 2.5, -3.25, 10.0, 0.5, 100.0],
        &[2, 3],
        "┌→────────────┐\n↓ 1 2.5  -3.25│\n│10 0.5 100   │\n└~────────────┘",
    );
    // Widths count chars, not bytes; text left-aligns.
    assert_display_golden::<&str, U2>(
        "multibyte",
        &["█", "░", "♣ ace", "é"],
        &[2, 2],
        "┌→──────┐\n↓█     ░│\n│♣ ace é│\n└~──────┘",
    );
}

#[test]
fn display_forwards_precision_and_width() {
    let a = new_arr::<f64, U1>(&[1.0 / 3.0, 2.0 / 3.0], &[2]);
    assert_eq!(format!("{a:.2}"), "┌→────────┐\n│0.33 0.67│\n└~────────┘");
    let b = new_arr::<i32, U1>(&[1, 2], &[2]);
    assert_eq!(format!("{b:>3}"), "┌→──────┐\n│  1   2│\n└~──────┘");
    let z = new_arr::<C<f64>, U1>(&[C(1.25, -0.5)], &[1]);
    assert_eq!(format!("{z:.1}"), "┌→───────┐\n│1.2-0.5i│\n└~───────┘");
}

#[test]
fn display_nests_arrays_as_blocks() {
    let ragged = Ndarr::from(vec![
        Ndarr::from(vec![1, 2]),
        Ndarr::from(vec![3, 4, 5]),
        Ndarr::from(vec![6]),
    ]);
    assert_eq!(
        format!("{ragged}"),
        "┌→────────────────┐\n│┌→──┐ ┌→────┐ ┌→┐│\n││1 2│ │3 4 5│ │6││\n│└~──┘ └~────┘ └~┘│\n└∊────────────────┘"
    );
}

// Debug drops the private offset/strides view metadata (pinned in behavior.rs).
#[test]
fn display_debug_golden() {
    let data: Vec<i32> = (0..6).collect();
    let n = new_arr::<i32, U2>(&data, &[2, 3]);
    let dn = format!("{:?}", n);
    assert!(
        dn.starts_with("Ndarr { data: [0, 1, 2, 3, 4, 5], dim: Dim { shape: [2, 3] }"),
        "{dn}"
    );
}

#[test]
fn display_noncontiguous_matches_materialized() {
    let data: Vec<i32> = (0..24)
        .map(|i| if i % 5 == 0 { -i * 37 } else { i })
        .collect();
    let n = new_arr::<i32, U3>(&data, &[2, 3, 4]);

    let tv = n.t_view();
    assert_eq!(
        format!("{}", tv),
        format!("{}", n.t()),
        "t_view Display != materialized t Display"
    );

    for axis in 0..3 {
        for i in 0..n.shape()[axis] {
            let v = n.index_axis_view(axis, i).unwrap();
            let owned = v.to_owned_array();
            assert_eq!(
                format!("{}", v),
                format!("{}", owned),
                "index_axis_view({axis},{i}) Display != materialized"
            );
        }
    }

    let row: Vec<i32> = vec![1, -22, 333, -4444];
    let r = new_arr::<i32, U1>(&row, &[4]);
    let target = Dim::<U3>::new(&[2, 3, 4]).unwrap();
    let bv = r.broadcast_view_to(&target).unwrap();
    assert_eq!(
        format!("{}", bv),
        format!("{}", bv.to_owned_array()),
        "broadcast_view_to Display != materialized"
    );

    let big: Vec<i32> = (0..(20 * 21))
        .map(|i| if i % 7 == 0 { -i } else { i })
        .collect();
    let bg = new_arr::<i32, U2>(&big, &[20, 21]);
    assert_eq!(
        format!("{}", bg.t_view()),
        format!("{}", bg.t()),
        "collapsed t_view Display != materialized"
    );

    let inner = n.index_axis_view(1, 2).unwrap();
    assert_eq!(
        format!("{}", inner.t_view()),
        format!("{}", inner.to_owned_array().t()),
        "t_view of axis view Display != materialized"
    );
}

#[test]
fn display_random_noncontiguous_matches_materialized() {
    let mut g = Prng::new();
    for trial in 0..N_RANDOM {
        let shape = [g.usize_in(1, 4), g.usize_in(1, 4), g.usize_in(1, 4)];
        let n: usize = shape.iter().product();
        let data: Vec<i32> = (0..n).map(|_| g.f64_in(-9999.0, 9999.0) as i32).collect();
        let a = new_arr::<i32, U3>(&data, &shape);
        assert_eq!(
            format!("{}", a.t_view()),
            format!("{}", a.t()),
            "rand_t_view[{trial}] shape={shape:?}"
        );
        let axis = g.usize_in(0, 2);
        let idx = g.usize_in(0, shape[axis] - 1);
        let v = a.index_axis_view(axis, idx).unwrap();
        assert_eq!(
            format!("{}", v),
            format!("{}", v.to_owned_array()),
            "rand_axis_view[{trial}] shape={shape:?} axis={axis} idx={idx}"
        );
    }
}

#[test]
fn error_paths_return_err() {
    let data: Vec<i32> = (0..6).collect();
    let a = new_arr::<i32, U2>(&data, &[2, 3]);

    assert!(Ndarr::<i32, U2>::new(&data, [2, 4]).is_err());

    assert!(a.view().reshape([4usize, 4]).is_err());

    assert!(a.flat_index(&[2, 0]).is_err());
    assert!(a.flat_index(&[0]).is_err());
    assert!(a.flat_index(&[0, 3]).is_err());

    assert!(a.index_axis_view(2, 0).is_err());
    assert!(a.index_axis_view(0, 2).is_err());
    assert!(a.index_axis_view(5, 0).is_err());

    let too_small = Dim::<U1>::new(&[3]).unwrap();
    assert!(a.broadcast_view_to(&too_small).is_err());
    let incompatible = Dim::<U2>::new(&[5, 5]).unwrap();
    assert!(a.broadcast_view_to(&incompatible).is_err());
}

#[test]
fn zero_axis_and_degenerate_shapes() {
    let a = new_arr::<i32, U2>(&[], &[0, 3]);
    assert_eq!(a.shape(), &[0, 3]);
    assert_eq!(a.len(), 0);
    assert!(a.is_empty());
    assert_eq!(a.iter_elems().count(), 0);
    assert_eq!(a.to_owned_array().data(), &[] as &[i32]);

    let c = new_arr::<i32, U3>(&[], &[2, 0, 3]);
    assert_eq!(c.len(), 0);
    assert_eq!(c.t_view().to_owned_array().shape(), &[3, 0, 2]);

    let e = new_arr::<i32, U3>(&[11], &[1, 1, 1]);
    assert_eq!(e[[0, 0, 0]], 11);
    assert_eq!(format!("{}", e), "┌┌→─┐\n↓↓11│\n└└~─┘");
}

#[test]
fn complex_scalar_ops_both_orders() {
    for ((ar, ai), (br, bi)) in probe_points()
        .into_iter()
        .zip(probe_points().into_iter().rev())
    {
        let an = C(ar, ai);
        let bn = C(br, bi);

        assert!(
            approx_c64(an + bn, C(ar + br, ai + bi), EPS_64),
            "add {an:?} {bn:?}"
        );
        assert!(approx_c64(bn + an, C(br + ar, bi + ai), EPS_64), "add rev");
        assert!(approx_c64(an - bn, C(ar - br, ai - bi), EPS_64), "sub");
        assert!(approx_c64(bn - an, C(br - ar, bi - ai), EPS_64), "sub rev");
        assert!(
            approx_c64(an * bn, cp(rmul((ar, ai), (br, bi))), EPS_64),
            "mul"
        );
        assert!(
            approx_c64(bn * an, cp(rmul((br, bi), (ar, ai))), EPS_64),
            "mul rev"
        );
        assert!(
            approx_c64(an / bn, cp(rdiv((ar, ai), (br, bi))), EPS_64),
            "div"
        );
        assert!(
            approx_c64(bn / an, cp(rdiv((br, bi), (ar, ai))), EPS_64),
            "div rev"
        );
        assert!(approx_c64(-an, C(-ar, -ai), EPS_64), "neg");
        assert!(approx_c64(an.conj(), C(ar, -ai), EPS_64), "conj");
        // inv = conj(z) / |z|^2
        let r_sq = ar * ar + ai * ai;
        assert!(
            approx_c64(an.inv(), C(ar / r_sq, -ai / r_sq), EPS_64),
            "inv"
        );
        assert_eq!(an.r_square(), r_sq, "r_square");
        assert_eq!(an.re(), ar);
        assert_eq!(an.im(), ai);

        assert!(approx_c64(an + bn, an + bn, EPS_64), "ref add");
        assert!(approx_c64(an * bn, an * bn, EPS_64), "ref/ref mul");
        assert!(approx_c64(an - bn, an - bn, EPS_64), "val/ref sub");
        assert!(approx_c64(-&an, -an, EPS_64), "ref neg");

        // powi(k) == repeated multiplication (k>0), 1/product (k<0), 1 (k=0)
        for k in [-3i32, -1, 0, 1, 2, 5] {
            let expected = if k == 0 {
                (1.0, 0.0)
            } else {
                let mut acc = (ar, ai);
                for _ in 1..k.abs() {
                    acc = rmul(acc, (ar, ai));
                }
                if k > 0 {
                    acc
                } else {
                    rdiv((1.0, 0.0), acc)
                }
            };
            assert!(
                approx_c64(an.powi(k), cp(expected), EPS_64),
                "powi({k}) of {an:?}: got {:?} expected {expected:?}",
                an.powi(k)
            );
        }
    }
}

#[test]
fn complex_real_mixed_ops() {
    for (r, i) in probe_points() {
        let n = C(r, i);
        let s = 2.5f64;
        assert!(approx_c64(n + s, C(r + s, i), EPS_64), "C+real");
        assert!(approx_c64(s + n, C(s + r, i), EPS_64), "real+C");
        assert!(approx_c64(n - s, C(r - s, i), EPS_64), "C-real");
        assert!(approx_c64(s - n, C(s - r, -i), EPS_64), "real-C");
        assert!(approx_c64(n * s, C(r * s, i * s), EPS_64), "C*real");
        assert!(approx_c64(s * n, C(s * r, s * i), EPS_64), "real*C");
        assert!(approx_c64(n / s, C(r / s, i / s), EPS_64), "C/real");
        assert!(
            approx_c64(s / n, cp(rdiv((s, 0.0), (r, i))), EPS_64),
            "real/C {n:?}"
        );

        let mut mn = n;
        mn += s;
        mn -= 1.0;
        mn *= 3.0;
        mn /= 2.0;
        mn += C(1.0, 1.0);
        mn -= C(0.5, 0.5);
        mn *= C(0.0, 1.0);
        mn /= C(2.0, 0.0);
        let mut e = (r, i);
        e = (e.0 + s, e.1);
        e = (e.0 - 1.0, e.1);
        e = (e.0 * 3.0, e.1 * 3.0);
        e = (e.0 / 2.0, e.1 / 2.0);
        e = radd(e, (1.0, 1.0));
        e = rsub(e, (0.5, 0.5));
        e = rmul(e, (0.0, 1.0));
        e = rdiv(e, (2.0, 0.0));
        assert!(approx_c64(mn, cp(e), EPS_64), "assign chain {n:?}");
    }

    let lit: C<i32> = 1 + 2.i();
    assert_eq!((lit.re(), lit.im()), (1, 2));
    let zn: C<i32> = 3.i();
    assert_eq!((zn.re(), zn.im()), (0, 3));
}

#[test]
fn complex_display_golden() {
    let cases: Vec<(C<f64>, &str, &str)> = vec![
        (C(1.0, 1.0), "1+1i", "C(1.0, 1.0)"),
        (C(-1.0, 1.0), "-1+1i", "C(-1.0, 1.0)"),
        (C(-1.0, -1.0), "-1-1i", "C(-1.0, -1.0)"),
        (C(1.0, -1.0), "1-1i", "C(1.0, -1.0)"),
        (C(3.5, 0.25), "3.5+0.25i", "C(3.5, 0.25)"),
        (C(-3.5, -0.25), "-3.5-0.25i", "C(-3.5, -0.25)"),
        (C(0.0, 1.0), "0+1i", "C(0.0, 1.0)"),
        (C(0.0, -1.0), "0-1i", "C(0.0, -1.0)"),
        (C(2.0, 0.0), "2+0i", "C(2.0, 0.0)"),
        (C(-2.5, 0.0), "-2.5+0i", "C(-2.5, 0.0)"),
        (C(-2.5, -0.0), "-2.5-0i", "C(-2.5, -0.0)"),
        (C(0.0, 0.0), "0+0i", "C(0.0, 0.0)"),
        (C(-0.0, 0.0), "-0+0i", "C(-0.0, 0.0)"),
    ];
    for (z, disp, dbg) in cases {
        assert_eq!(format!("{}", z), disp, "C<f64> Display {z:?}");
        assert_eq!(format!("{:?}", z), dbg, "C<f64> Debug {z:?}");
    }
    let int_cases: Vec<(C<i32>, &str)> = vec![
        (C(1, 2), "1+2i"),
        (C(-1, -2), "-1-2i"),
        (C(0, 0), "0+0i"),
        (C(5, 0), "5+0i"),
        (C(0, -7), "0-7i"),
    ];
    for (z, disp) in int_cases {
        assert_eq!(format!("{}", z), disp, "C<i32> Display {z:?}");
    }
}

#[test]
fn complex_from_real_embeds_as_imaginary_zero() {
    for (r, _) in probe_points() {
        let z: C<f64> = C::from(r);
        assert!(approx_c64(z, C(r, 0.0), EPS_64), "From<f64>");
    }
    assert_eq!(C::from(42u8), C(42, 0));
    assert_eq!(C::from(-3i32), C(-3, 0));
}

#[test]
fn complex_float_fns_quadrants_and_branch_cuts() {
    let pn = probe_c64();
    macro_rules! cmp {
        ($name:literal, $f:expr, $reff:expr) => {{
            let got: Vec<C<f64>> = pn.iter().map($f).collect();
            let expected: Vec<C<f64>> = pn.iter().map(|z| cp($reff(pc(z)))).collect();
            assert_c64_slices($name, &got, &expected);
        }};
    }
    cmp!("exp", |z| z.exp(), rexp);
    cmp!("ln", |z| z.ln(), rln);
    cmp!("sqrt", |z| z.sqrt(), rsqrt);
    cmp!("sin", |z| z.sin(), rsin);
    cmp!("cos", |z| z.cos(), rcos);
    cmp!("tan", |z| z.tan(), rtan);
    cmp!("sinh", |z| z.sinh(), rsinh);
    cmp!("cosh", |z| z.cosh(), rcosh);
    cmp!("tanh", |z| z.tanh(), rtanh);
    cmp!("powf_2", |z| z.powf(2.0), |p| rpowf(p, 2.0));
    cmp!("powf_neg", |z| z.powf(-1.5), |p| rpowf(p, -1.5));
    cmp!("powf_zero", |z| z.powf(0.0), |p| rpowf(p, 0.0));
    cmp!("conj", |z| z.conj(), |p: P| (p.0, -p.1));
    cmp!("powc", |z| z.powc(C(0.5, -0.25)), |p| rpowc(
        p,
        (0.5, -0.25)
    ));

    let got_abs: Vec<f64> = pn.iter().map(|z| z.abs()).collect();
    let exp_abs: Vec<f64> = pn.iter().map(|z| rabs(pc(z))).collect();
    assert_f64_slices("abs", &got_abs, &exp_abs);
    let got_arg: Vec<f64> = pn.iter().map(|z| z.arg()).collect();
    let exp_arg: Vec<f64> = pn.iter().map(|z| rarg(pc(z))).collect();
    assert_f64_slices("arg", &got_arg, &exp_arg);

    for z in pn.iter() {
        let (rn, tn) = z.to_polar();
        assert_f64_slices("to_polar", &[rn, tn], &[rabs(pc(z)), rarg(pc(z))]);
        // from_polar(r, theta) == r * e^{i theta}
        assert!(
            approx_c64(
                C::from_polar(rn, tn),
                cp((rn * tn.cos(), rn * tn.sin())),
                EPS_64
            ),
            "from_polar {z:?}"
        );
        assert_eq!(z.is_nan(), z.0.is_nan() || z.1.is_nan(), "is_nan {z:?}");
        assert_eq!(
            z.is_finite(),
            z.0.is_finite() && z.1.is_finite(),
            "is_finite {z:?}"
        );
        assert_eq!(
            z.is_infinite(),
            z.0.is_infinite() || z.1.is_infinite(),
            "is_infinite {z:?}"
        );
        assert_eq!(
            z.is_normal(),
            z.0.is_normal() && z.1.is_normal(),
            "is_normal {z:?}"
        );
    }

    // sign of zero picks the sqrt branch: arg(-4 + 0i) = pi, arg(-4 - 0i) = -pi.
    let s_pos = C(-4.0, 0.0).sqrt();
    assert!(
        same_f64(s_pos.0, 0.0, EPS_64) && same_f64(s_pos.1, 2.0, EPS_64),
        "{s_pos:?}"
    );
    let s_neg = C(-4.0, -0.0).sqrt();
    assert!(
        same_f64(s_neg.0, 0.0, EPS_64) && same_f64(s_neg.1, -2.0, EPS_64),
        "{s_neg:?}"
    );
}

#[test]
fn complex_float_fns_f32() {
    // f32 references: f64 oracle rounded once at the end; EPS_32 absorbs the difference.
    let pn: Vec<C<f32>> = probe_points()
        .into_iter()
        .map(|(r, i)| C(r as f32, i as f32))
        .collect();
    macro_rules! cmp32 {
        ($name:literal, $f:expr, $reff:expr) => {{
            let got: Vec<C<f32>> = pn.iter().map($f).collect();
            let expected: Vec<C<f32>> = pn
                .iter()
                .map(|z| {
                    let r = $reff((z.0 as f64, z.1 as f64));
                    C(r.0 as f32, r.1 as f32)
                })
                .collect();
            assert_c32_slices($name, &got, &expected);
        }};
    }
    cmp32!("exp32", |z| z.exp(), rexp);
    cmp32!("ln32", |z| z.ln(), rln);
    cmp32!("sqrt32", |z| z.sqrt(), rsqrt);
    cmp32!("sin32", |z| z.sin(), rsin);
    cmp32!("cos32", |z| z.cos(), rcos);
    cmp32!("powf32", |z| z.powf(1.5), |p| rpowf(p, 1.5));
}

fn complex_arr<R: StaticRank>(reals: &[f64], shape: &[usize]) -> Ndarr<C<f64>, R> {
    let data: Vec<C<f64>> = reals
        .chunks(2)
        .map(|c| C(c[0], *c.get(1).unwrap_or(&0.0)))
        .collect();
    Ndarr::new(&data, Dim::<R>::new(shape).unwrap()).unwrap()
}

#[test]
fn complex_tensor_elementwise() {
    let flat: Vec<f64> = (0..24).map(|i| (i as f64) * 0.5 - 3.0).collect();
    let a = complex_arr::<U2>(&flat, &[3, 4]);
    let elems: Vec<C<f64>> = a.data().to_vec();

    let exp_re: Vec<f64> = elems.iter().map(|z| z.0).collect();
    assert_f64_slices("re", a.re().data(), &exp_re);
    let exp_im: Vec<f64> = elems.iter().map(|z| z.1).collect();
    assert_f64_slices("im", a.im().data(), &exp_im);
    let exp_conj: Vec<C<f64>> = elems.iter().map(|z| z.conj()).collect();
    assert_c64_slices("conj", a.conj().data(), &exp_conj);

    // h() == conj + transpose
    let h = a.h();
    assert_eq!(h.shape(), &[4, 3]);
    for r in 0..3 {
        for c in 0..4 {
            let z = a[[r, c]];
            assert!(approx_c64(h[[c, r]], z.conj(), EPS_64), "h at ({r},{c})");
        }
    }

    let exp_inv: Vec<C<f64>> = elems.iter().map(|z| z.inv()).collect();
    assert_c64_slices("inv", a.inv().data(), &exp_inv);
    let exp_rsq: Vec<f64> = elems.iter().map(|z| z.r_square()).collect();
    assert_f64_slices("r_square", a.r_square().data(), &exp_rsq);
    for k in [-2i32, 0, 1, 3] {
        let expected: Vec<C<f64>> = elems.iter().map(|z| z.powi(k)).collect();
        assert_c64_slices(&format!("powi({k})"), a.powi(k).data(), &expected);
    }

    let reals: Vec<f64> = (0..12).map(|i| i as f64 - 6.0).collect();
    let rn = new_arr::<f64, U3>(&reals, &[2, 2, 3]);
    let exp_cx: Vec<C<f64>> = reals.iter().map(|&x| C(x, 0.0)).collect();
    assert_c64_slices("to_complex", rn.to_complex().data(), &exp_cx);

    let ints: Vec<i32> = (0..6).collect();
    let in_ = new_arr::<i32, U2>(&ints, &[2, 3]);
    let cn = in_.to_complex();
    assert_eq!(cn.shape(), &[2, 3]);
    for (x, y) in cn.data().iter().zip(ints.iter()) {
        assert_eq!((x.re(), x.im()), (*y, 0), "to_complex i32");
    }
}

#[test]
fn complex_tensor_float_fns() {
    let flat: Vec<f64> = (0..32).map(|i| (i as f64) * 0.375 - 6.0).collect();
    let a = complex_arr::<U2>(&flat, &[4, 4]);
    let elems: Vec<C<f64>> = a.data().to_vec();

    macro_rules! cmp_t {
        ($name:literal, $t:expr, $s:expr) => {{
            let expected: Vec<C<f64>> = elems.iter().map($s).collect();
            assert_c64_slices($name, $t.data(), &expected);
        }};
    }
    let exp_abs: Vec<f64> = elems.iter().map(|z| z.abs()).collect();
    assert_f64_slices("t_abs", a.abs().data(), &exp_abs);
    let exp_arg: Vec<f64> = elems.iter().map(|z| z.arg()).collect();
    assert_f64_slices("t_arg", a.arg().data(), &exp_arg);
    cmp_t!("t_exp", a.exp(), |z| z.exp());
    cmp_t!("t_ln", a.ln(), |z| z.ln());
    cmp_t!("t_sqrt", a.sqrt(), |z| z.sqrt());
    cmp_t!("t_sin", a.sin(), |z| z.sin());
    cmp_t!("t_cos", a.cos(), |z| z.cos());
    cmp_t!("t_tan", a.tan(), |z| z.tan());
    cmp_t!("t_powf", a.powf(1.75), |z| z.powf(1.75));
    cmp_t!("t_powc", a.powc(C(0.5, 0.5)), |z| z.powc(C(0.5, 0.5)));

    let exp_nan: Vec<bool> = elems.iter().map(|z| z.is_nan()).collect();
    assert_eq!(a.is_nan().data(), exp_nan.as_slice());
    let exp_fin: Vec<bool> = elems.iter().map(|z| z.is_finite()).collect();
    assert_eq!(a.is_finite().data(), exp_fin.as_slice());
    let exp_inf: Vec<bool> = elems.iter().map(|z| z.is_infinite()).collect();
    assert_eq!(a.is_infinite().data(), exp_inf.as_slice());
    let exp_norm: Vec<bool> = elems.iter().map(|z| z.is_normal()).collect();
    assert_eq!(a.is_normal().data(), exp_norm.as_slice());

    let pn = a.to_polar();
    assert_eq!(pn.shape(), &[4, 4]);
    for (i, (x, z)) in pn.data().iter().zip(elems.iter()).enumerate() {
        let (er, et) = z.to_polar();
        assert_f64_slices(&format!("t_to_polar[{i}]"), &[x.0, x.1], &[er, et]);
    }
}

#[test]
fn complex_tensor_arithmetic_both_orders() {
    let flat: Vec<f64> = (0..12).map(|i| i as f64 - 5.0).collect();
    let a = complex_arr::<U2>(&flat, &[2, 3]);
    let flat2: Vec<f64> = (0..12).map(|i| 3.0 - (i as f64) * 0.5).collect();
    let c = complex_arr::<U2>(&flat2, &[2, 3]);
    let av: Vec<C<f64>> = a.data().to_vec();
    let cv: Vec<C<f64>> = c.data().to_vec();

    let zip = |f: fn(C<f64>, C<f64>) -> C<f64>| -> Vec<C<f64>> {
        av.iter().zip(cv.iter()).map(|(x, y)| f(*x, *y)).collect()
    };
    assert_c64_slices("cadd", (&a + &c).data(), &zip(|x, y| x + y));
    assert_c64_slices("cadd_rev", (&c + &a).data(), &zip(|x, y| y + x));
    assert_c64_slices("csub", (&a - &c).data(), &zip(|x, y| x - y));
    assert_c64_slices("csub_rev", (&c - &a).data(), &zip(|x, y| y - x));
    assert_c64_slices("cmul", (&a * &c).data(), &zip(|x, y| x * y));
    assert_c64_slices("cmul_rev", (&c * &a).data(), &zip(|x, y| y * x));
    assert_c64_slices("cdiv", (&a / &c).data(), &zip(|x, y| x / y));

    let reals: Vec<f64> = (0..6).map(|i| i as f64 + 1.0).collect();
    let rn = new_arr::<f64, U2>(&reals, &[2, 3]);
    let exp1: Vec<C<f64>> = reals.iter().map(|&x| C(x + 1.0, 1.0)).collect();
    assert_c64_slices("real_arr_plus_c", (&rn + C(1.0, 1.0)).data(), &exp1);
    let exp2: Vec<C<f64>> = av.iter().map(|z| *z + 2.0).collect();
    assert_c64_slices("c_arr_plus_real_scalar", (&a + 2.0).data(), &exp2);
    let exp3: Vec<C<f64>> = av.iter().map(|z| *z * C(0.0, 1.0)).collect();
    assert_c64_slices("c_arr_times_i", (&a * 1.0.i()).data(), &exp3);

    let row: Vec<f64> = vec![1.0, -1.0, 2.0, -2.0, 3.0, -3.0];
    let vn = complex_arr::<U1>(&row, &[3]);
    let vv: Vec<C<f64>> = vn.data().to_vec();
    let exp_b: Vec<C<f64>> = (0..6).map(|i| av[i] + vv[i % 3]).collect();
    assert_c64_slices("cbroadcast", (&a + &vn).data(), &exp_b);
    assert_c64_slices("cbroadcast_rev", (&vn + &a).data(), &exp_b);
}

#[test]
fn complex_tensor_noncontiguous_inputs() {
    let flat: Vec<f64> = (0..48).map(|i| (i as f64) * 0.25 - 6.0).collect();
    let a = complex_arr::<U3>(&flat, &[2, 3, 4]);

    let ta = a.t_view().to_owned_array();
    assert_eq!(ta.shape(), &[4, 3, 2]);
    let tv: Vec<C<f64>> = ta.data().to_vec();
    let map_c = |f: fn(&C<f64>) -> C<f64>| -> Vec<C<f64>> { tv.iter().map(f).collect() };
    assert_c64_slices("nc_conj", ta.conj().data(), &map_c(|z| z.conj()));
    assert_c64_slices("nc_sqrt", ta.sqrt().data(), &map_c(|z| z.sqrt()));
    assert_c64_slices("nc_ln", ta.ln().data(), &map_c(|z| z.ln()));
    let exp_abs: Vec<f64> = tv.iter().map(|z| z.abs()).collect();
    assert_f64_slices("nc_abs", ta.abs().data(), &exp_abs);
    let exp_re: Vec<f64> = tv.iter().map(|z| z.re()).collect();
    assert_f64_slices("nc_re", ta.re().data(), &exp_re);
    assert_c64_slices("nc_inv", ta.inv().data(), &map_c(|z| z.inv()));

    let view = a.t_view();
    assert_c64_slices("view_conj", view.conj().data(), &map_c(|z| z.conj()));
    assert_c64_slices("view_sqrt", view.sqrt().data(), &map_c(|z| z.sqrt()));
    assert_f64_slices("view_abs", view.abs().data(), &exp_abs);
    assert_f64_slices("view_re", view.re().data(), &exp_re);

    for x in 0..4 {
        for y in 0..3 {
            for z in 0..2 {
                assert!(
                    approx_c64(ta[[x, y, z]], a[[z, y, x]], EPS_64),
                    "transpose data at ({x},{y},{z})"
                );
            }
        }
    }

    for axis in 0..3 {
        let va = a.index_axis_view(axis, 1).unwrap().to_owned_array();
        let vd: Vec<C<f64>> = va.data().to_vec();
        let exp_conj: Vec<C<f64>> = vd.iter().map(|z| z.conj()).collect();
        assert_c64_slices(&format!("nc_axis{axis}_conj"), va.conj().data(), &exp_conj);
        let exp_exp: Vec<C<f64>> = vd.iter().map(|z| z.exp()).collect();
        assert_c64_slices(&format!("nc_axis{axis}_exp"), va.exp().data(), &exp_exp);
    }

    let reals: Vec<f64> = vec![1.5, -2.5, 3.5, -4.5];
    let rn = new_arr::<f64, U1>(&reals, &[4]);
    let dim = Dim::<U3>::new(&[2, 3, 4]).unwrap();
    let bn = rn.broadcast_view_to(&dim).unwrap().to_owned_array();
    let exp_bcx: Vec<C<f64>> = (0..24).map(|i| C(reals[i % 4], 0.0)).collect();
    assert_c64_slices("nc_to_complex", bn.to_complex().data(), &exp_bcx);
    let exp_bconj: Vec<C<f64>> = exp_bcx.iter().map(|z| z.conj()).collect();
    assert_c64_slices(
        "nc_broadcast_conj",
        bn.to_complex().conj().data(),
        &exp_bconj,
    );

    assert_eq!(
        format!("{}", a.t_view()),
        format!("{}", ta),
        "complex t_view Display != materialized"
    );
}

#[test]
fn complex_tensor_rank0_and_rank1_edges() {
    let a = complex_arr::<U0>(&[2.0, -3.0], &[]);
    assert_eq!(a.rank(), 0);
    assert_c64_slices("rank0_conj", a.conj().data(), &[C(2.0, -3.0).conj()]);
    assert_c64_slices("rank0_sqrt", a.sqrt().data(), &[C(2.0, -3.0).sqrt()]);
    assert_eq!(format!("{}", a), format!("{}", C(2.0, -3.0)));

    let c = complex_arr::<U1>(&[1.0, 1.0], &[1]);
    assert_c64_slices("rank1_len1_ln", c.ln().data(), &[C(1.0, 1.0).ln()]);
    assert_c64_slices("rank1_len1_h", c.h().data(), &[C(1.0, 1.0).conj()]);
    assert_eq!(format!("{}", c), "┌→───┐\n│1+1i│\n└~───┘");

    let en: Ndarr<C<f64>, U1> = Ndarr::new(&[], Dim::<U1>::new(&[0]).unwrap()).unwrap();
    assert_eq!(en.conj().data().len(), 0);
    assert_eq!(en.re().data().len(), 0);
}

#[test]
fn complex_random_differential() {
    let mut g = Prng::new();
    for trial in 0..N_RANDOM {
        let shape = [g.usize_in(1, 4), g.usize_in(1, 4)];
        let n: usize = shape.iter().product();
        let flat: Vec<f64> = (0..n * 2).map(|_| g.f64_in(-6.0, 6.0)).collect();
        let a = complex_arr::<U2>(&flat, &shape);
        let flat2: Vec<f64> = (0..n * 2).map(|_| g.f64_in(-6.0, 6.0)).collect();
        let c = complex_arr::<U2>(&flat2, &shape);
        let av: Vec<C<f64>> = a.data().to_vec();
        let cv: Vec<C<f64>> = c.data().to_vec();

        let tag = |s: &str| format!("rand[{trial}]{s} shape={shape:?}");
        let zip = |f: fn(C<f64>, C<f64>) -> C<f64>| -> Vec<C<f64>> {
            av.iter().zip(cv.iter()).map(|(x, y)| f(*x, *y)).collect()
        };
        assert_c64_slices(&tag(".add"), (&a + &c).data(), &zip(|x, y| x + y));
        assert_c64_slices(&tag(".mul"), (&a * &c).data(), &zip(|x, y| x * y));
        assert_c64_slices(&tag(".sub_rev"), (&c - &a).data(), &zip(|x, y| y - x));

        let map_c = |f: fn(&C<f64>) -> C<f64>| -> Vec<C<f64>> { av.iter().map(f).collect() };
        assert_c64_slices(&tag(".exp"), a.exp().data(), &map_c(|z| z.exp()));
        assert_c64_slices(&tag(".ln"), a.ln().data(), &map_c(|z| z.ln()));
        assert_c64_slices(&tag(".sqrt"), a.sqrt().data(), &map_c(|z| z.sqrt()));
        assert_c64_slices(&tag(".sin"), a.sin().data(), &map_c(|z| z.sin()));
        assert_c64_slices(&tag(".tan"), a.tan().data(), &map_c(|z| z.tan()));
        assert_c64_slices(&tag(".conj"), a.conj().data(), &map_c(|z| z.conj()));
        assert_c64_slices(&tag(".inv"), a.inv().data(), &map_c(|z| z.inv()));
        assert_c64_slices(&tag(".powi3"), a.powi(3).data(), &map_c(|z| z.powi(3)));
        let exp_abs: Vec<f64> = av.iter().map(|z| z.abs()).collect();
        assert_f64_slices(&tag(".abs"), a.abs().data(), &exp_abs);
        let exp_arg: Vec<f64> = av.iter().map(|z| z.arg()).collect();
        assert_f64_slices(&tag(".arg"), a.arg().data(), &exp_arg);

        let exp_ref: Vec<C<f64>> = av.iter().map(|z| cp(rexp(pc(z)))).collect();
        assert_c64_slices(&tag(".exp_ref"), a.exp().data(), &exp_ref);
        let mul_ref: Vec<C<f64>> = av
            .iter()
            .zip(cv.iter())
            .map(|(x, y)| cp(rmul(pc(x), pc(y))))
            .collect();
        assert_c64_slices(&tag(".mul_ref"), (&a * &c).data(), &mul_ref);

        let ta = a.t_view().to_owned_array();
        let td: Vec<C<f64>> = ta.data().to_vec();
        let exp_tconj: Vec<C<f64>> = td.iter().map(|z| z.conj()).collect();
        assert_c64_slices(&tag(".t_conj"), ta.conj().data(), &exp_tconj);
        let exp_tln: Vec<C<f64>> = td.iter().map(|z| z.ln()).collect();
        assert_c64_slices(&tag(".t_ln"), ta.ln().data(), &exp_tln);
        assert_eq!(
            format!("{}", a.t_view()),
            format!("{}", ta),
            "{}",
            tag(".t_display")
        );
    }
}

#[test]
fn complex_tensor_powc_uses_its_argument() {
    // Ndarr::<C<T>>::powc must compute z^w with the given exponent, not z^z.
    let flat: Vec<f64> = vec![2.0, 0.5, 1.0, 1.0, -1.0, 0.25];
    let a = complex_arr::<U1>(&flat, &[3]);
    let w = C(0.5, 0.0);
    let got = a.powc(w);
    let expected: Vec<C<f64>> = a.data().iter().map(|z| z.powc(w)).collect();
    assert_c64_slices("powc", got.data(), &expected);
    // and differs from z^z for the first element
    let z_pow_z = a.data()[0].powc(a.data()[0]);
    assert!(
        !approx_c64(got.data()[0], z_pow_z, EPS_64),
        "tensor powc still computes z^z"
    );
}

#[test]
fn real_divided_by_complex_is_correct() {
    // real / C<T> divides by |rhs|², so 4/(2+0i) == 2+0i.
    let n = 4.0f64 / C(2.0f64, 0.0);
    assert!(approx_c64(n, C(2.0, 0.0), EPS_64), "real/C: {n:?}");
    let m = 3.0f64 / C(1.0f64, 2.0);
    assert!(approx_c64(m, C(0.6, -1.2), EPS_64), "real/C: {m:?}");
}

#[test]
fn complex_is_infinite_is_not_is_finite() {
    // is_infinite is true iff either component is infinite.
    assert!(C(f64::INFINITY, 0.0).is_infinite());
    assert!(C(0.0, f64::NEG_INFINITY).is_infinite());
    assert!(!C(1.0, 1.0).is_infinite());
    assert!(!C(f64::NAN, 0.0).is_infinite());
}
