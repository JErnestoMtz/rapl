use super::*;
use rand;
use rand::distributions::uniform::SampleUniform;
use rand::distributions::Uniform;
use rand::prelude::*;
use rand::seq::IteratorRandom;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Normal, StandardNormal};
use std::iter::IntoIterator;
/// Struct to encapsulate all the random functionality.
pub struct NdarrRand {}

impl NdarrRand {
    /// Generates a random N-dimensional array with values x ~ U(low, high)
    pub fn uniform<
        T: SampleUniform + Debug + Clone + Copy + Default,
        R: Unsigned,
        D: Into<Dim<R>>,
    >(
        low: T,
        high: T,
        shape: D,
        seed: Option<u64>,
    ) -> Ndarr<T, R> {
        let d: Dim<R> = shape.into();
        let n = d.get_number_elements();
        let random_vector: Vec<T> = match seed {
            Some(s) => {
                let mut rng = ChaCha8Rng::seed_from_u64(s);
                (0..n)
                    .map(|_| rng.sample(Uniform::new(low, high)))
                    .collect()
            }
            None => {
                let mut rng = rand::thread_rng();
                (0..n)
                    .map(|_| rng.sample(Uniform::new(low, high)))
                    .collect()
            }
        };
        Ndarr::new(&random_vector, d).unwrap()
    }

    /// Generates a random N-dimensional array with values x ~ N(μ, σ^2)
    pub fn normal<T: Debug + Clone + Copy + Default + Float, R: Unsigned, D: Into<Dim<R>>>(
        mu: T,
        sigma: T,
        shape: D,
        seed: Option<u64>,
    ) -> Ndarr<T, R>
    where
        StandardNormal: Distribution<T>,
    {
        let d: Dim<R> = shape.into();
        let n = d.get_number_elements();
        let normal = Normal::new(mu, sigma).unwrap();
        let random_vector: Vec<T> = match seed {
            Some(s) => {
                let mut rng = ChaCha8Rng::seed_from_u64(s);
                (0..n).map(|_| normal.sample(&mut rng)).collect()
            }
            None => {
                let mut rng = rand::thread_rng();
                (0..n).map(|_| normal.sample(&mut rng)).collect()
            }
        };
        Ndarr::new(&random_vector, d).unwrap()
    }

    /// Generate a random N-dimensional array by taking draws from a given array
    /// `elements` with replacement.
    pub fn choose<T: Debug + Clone + Default, R: Unsigned, D: Into<Dim<R>>>(
        elements: &[T],
        shape: D,
        seed: Option<u64>,
    ) -> Ndarr<T, R> 
    {
        let d: Dim<R> = shape.into();
        let n = d.get_number_elements();
        let elements = elements.iter();
        let random_vector: Vec<T> = match seed {
            Some(s) => {
                let mut rng = ChaCha8Rng::seed_from_u64(s);
                (0..n)
                    .map(|_| elements.clone().choose(&mut rng).unwrap().clone())
                    .collect()
            }
            None => {
                let mut rng = rand::thread_rng();
                (0..n)
                    .map(|_| elements.clone().choose(&mut rng).unwrap().clone())
                    .collect()
            }
        };
        Ndarr::new(&random_vector, d).unwrap()
    }
}

#[cfg(test)]
mod rng_arr {
    use super::*;
    #[test]
    fn test_normal_f64() {
        let arr = NdarrRand::normal(0f64, 1f64, [2, 2], Some(1234));
        let tgt = Ndarr::from([
            -0.3047064644834838,
            1.8246424684819205,
            0.4733072797360177,
            -0.717657616639252,
        ])
        .reshape([2, 2])
        .unwrap();
        assert!(arr == tgt);
    }

    #[test]
    fn test_normal_f32() {
        let arr = NdarrRand::normal(0f32, 1f32, [2, 2], Some(1234));
        let tgt = Ndarr::from([-0.30470645, 1.8246424, 0.47330728, -0.7176576])
            .reshape([2, 2])
            .unwrap();
        assert!(arr == tgt);
    }

    #[test]
    fn test_uniform_f64() {
        let arr = NdarrRand::uniform(0f64, 1f64, [2, 2], Some(1234));
        let tgt = Ndarr::from([
            0.38637312192058193,
            0.9963256225585044,
            0.5968809870290679,
            0.3163402777023183,
        ])
        .reshape([2, 2])
        .unwrap();
        assert!(arr == tgt);
    }

    #[test]
    fn test_uniform_f32() {
        let arr = NdarrRand::uniform(0f32, 1f32, [2, 2], Some(1234));
        let tgt = Ndarr::from([0.7023206, 0.38637304, 0.055616498, 0.9963256])
            .reshape([2, 2])
            .unwrap();
        assert!(arr == tgt);
    }

    #[test]
    fn test_choose() {
        let arr = NdarrRand::choose(&[1, 2, 3, 4, 5], [3, 3], Some(1234));
        let tgt = Ndarr::from([4, 1, 5, 2, 5, 4, 3, 3, 4])
            .reshape([3, 3])
            .unwrap();
        assert!(arr == tgt);
    }
}
