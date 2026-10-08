use super::{Ndarr, C, U1, U2};
use crate::buffer::Buffer;
use crate::Rank;
use num_traits::{Float, FromPrimitive};

/// e^(i · sign · 2π · num/den); the angle is computed in `f64` for accuracy.
fn twiddle<T: FromPrimitive>(num: usize, den: usize, inverse: bool) -> C<T> {
    let sign = if inverse { 1.0 } else { -1.0 };
    let (s, c) = (sign * std::f64::consts::TAU * num as f64 / den as f64).sin_cos();
    C(T::from_f64(c).unwrap(), T::from_f64(s).unwrap())
}

fn smallest_factor(n: usize) -> usize {
    (2..)
        .take_while(|p| p * p <= n)
        .find(|&p| n % p == 0)
        .unwrap_or(n)
}

/// DFT of any length by Cooley-Tukey decimation on the smallest prime factor,
/// with the O(n²) definition as the base case for prime lengths.
fn dft<T: Float + FromPrimitive>(x: &[C<T>], inverse: bool) -> Vec<C<T>> {
    let n = x.len();
    if n <= 1 {
        return x.to_vec();
    }
    let p = smallest_factor(n);
    if p == n {
        return (0..n)
            .map(|k| {
                x.iter()
                    .enumerate()
                    .fold(C(T::zero(), T::zero()), |acc, (j, v)| {
                        acc + *v * twiddle(j * k % n, n, inverse)
                    })
            })
            .collect();
    }
    // X[k] = Σ_r e^(i·sign·2π·rk/n) · DFT_m(x[r], x[r+p], ...)[k mod m]
    let m = n / p;
    let subs: Vec<Vec<C<T>>> = (0..p)
        .map(|r| {
            let seq: Vec<C<T>> = x[r..].iter().step_by(p).copied().collect();
            dft(&seq, inverse)
        })
        .collect();
    (0..n)
        .map(|k| {
            (0..p).fold(C(T::zero(), T::zero()), |acc, r| {
                acc + subs[r][k % m] * twiddle(r * k % n, n, inverse)
            })
        })
        .collect()
}

fn normalize<T: Float + FromPrimitive>(mut v: Vec<C<T>>, n: usize) -> Vec<C<T>> {
    let n_t = T::from_usize(n).unwrap();
    for z in v.iter_mut() {
        *z = C(z.0 / n_t, z.1 / n_t);
    }
    v
}

// Transforms gather elements in logical C-order and return owned arrays.
impl<T: Float + FromPrimitive, B: Buffer<C<T>>> Ndarr<C<T>, U1, B> {
    /// One dimensional Fourier transform.
    pub fn fft(&self) -> Ndarr<C<T>, U1> {
        let data: Vec<C<T>> = self.iter_elems().copied().collect();
        Ndarr::contiguous(dft(&data, false), self.dim.clone())
    }

    /// One dimensional inverse Fourier transform; `a.fft().ifft()` is approximately `a`.
    pub fn ifft(&self) -> Ndarr<C<T>, U1> {
        let data: Vec<C<T>> = self.iter_elems().copied().collect();
        Ndarr::contiguous(normalize(dft(&data, true), data.len()), self.dim.clone())
    }
}

impl<T: Float + FromPrimitive, B: Buffer<C<T>>> Ndarr<C<T>, U2, B> {
    /// Two dimensional Fourier transform, as a 1-D transform along each axis.
    pub fn fft2d(&self) -> Ndarr<C<T>, U2> {
        self.map_lanes(1, |lane| lane.fft())
            .expect("rank-2 arrays have axis 1")
            .map_lanes(0, |lane| lane.fft())
            .expect("rank-2 arrays have axis 0")
    }

    /// Two dimensional inverse Fourier transform; `a.fft2d().ifft2()` is approximately `a`.
    pub fn ifft2(&self) -> Ndarr<C<T>, U2> {
        self.map_lanes(1, |lane| lane.ifft())
            .expect("rank-2 arrays have axis 1")
            .map_lanes(0, |lane| lane.ifft())
            .expect("rank-2 arrays have axis 0")
    }
}

impl<T: Clone, R: Rank, B: Buffer<T>> Ndarr<T, R, B> {
    /// Shift the zero-frequency component to the center of the spectrum: a
    /// roll by `extent / 2` along every axis (NumPy's `fftshift`).
    pub fn fftshift(&self) -> Ndarr<T, R> {
        (0..self.rank()).fold(self.to_owned_array(), |shifted, axis| {
            shifted.roll((shifted.shape()[axis] / 2) as isize, axis)
        })
    }
}

#[cfg(test)]
mod fft_test {
    use super::*;
    #[test]
    fn test_1d() {
        let a = Ndarr::from([0.1, 0.2, 0.1, 0.0, 0.1, 0.0, 0.1, -0.1, 0.2]).to_complex();
        let fft_a_numpy: Ndarr<C<f64>, U1> = Ndarr::from([
            C(0.7, 0.0),
            C(0.26244852, -0.14456102),
            C(0.19606372, -0.09072781),
            C(-0.05, 0.08660254),
            C(-0.30851223, 0.31364084),
            C(-0.30851223, -0.31364084),
            C(-0.05, -0.08660254),
            C(0.19606372, 0.09072781),
            C(0.26244852, 0.14456102),
        ]);
        let rapl_fft = a.fft();

        assert!(rapl_fft.re().approx_epsilon(&fft_a_numpy.re(), 1e-8));
        assert!(rapl_fft.im().approx_epsilon(&fft_a_numpy.im(), 1e-8));
        assert!(rapl_fft.ifft().re().approx_epsilon(&a.re(), 1e-8));
        assert!(rapl_fft.ifft().im().approx_epsilon(&a.im(), 1e-8));
    }
    #[test]
    fn test_2d() {
        let a = Ndarr::from([0.1, 0.2, 0.1, 0.0, 0.1, 0.0, 0.1, -0.1, 0.2])
            .to_complex()
            .reshape([3, 3])
            .unwrap();
        let numpy_fft2: Ndarr<C<f64>, U2> = Ndarr::from([
            [C(0.7, 0.), C(-0.05, 0.08660254), C(-0.05, -0.08660254)],
            [
                C(0.25, 0.08660254),
                C(-0.35, -0.08660254),
                C(0.25, 0.25980762),
            ],
            [
                C(0.25, -0.08660254),
                C(0.25, -0.25980762),
                C(-0.35, 0.08660254),
            ],
        ]);
        let rapl_fft2 = a.fft2d();
        assert!(rapl_fft2.re().approx_epsilon(&numpy_fft2.re(), 1e-8));
        assert!(rapl_fft2.im().approx_epsilon(&numpy_fft2.im(), 1e-8));
        assert!(rapl_fft2.ifft2().re().approx_epsilon(&a.re(), 1e-8));
        assert!(rapl_fft2.ifft2().im().approx_epsilon(&a.im(), 1e-8));
    }

    #[test]
    fn fftshift_1d() {
        let odd = Ndarr::from([1, 2, 3, 4, 5, 6, 7]);
        let pair = Ndarr::from([1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(odd.fftshift(), Ndarr::from([5, 6, 7, 1, 2, 3, 4]));
        assert_eq!(pair.fftshift(), Ndarr::from([5, 6, 7, 8, 1, 2, 3, 4]));
    }

    #[test]
    fn fftshift_2d() {
        let odd = Ndarr::from(0..9).reshape([3, 3]).unwrap();
        let pair = Ndarr::from(0..16).reshape([4, 4]).unwrap();
        let odd_p = Ndarr::from(0..12).reshape([3, 4]).unwrap();
        assert_eq!(
            odd.fftshift(),
            Ndarr::from([[8, 6, 7], [2, 0, 1], [5, 3, 4]])
        );
        assert_eq!(
            pair.fftshift(),
            Ndarr::from([[10, 11, 8, 9], [14, 15, 12, 13], [2, 3, 0, 1], [6, 7, 4, 5]])
        );
        println!("{}", odd_p.fftshift());
    }
}
