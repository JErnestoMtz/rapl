use super::*;
use rand;
use rand::distributions::uniform::SampleUniform;
use rand::distributions::Uniform;
use rand::prelude::*;
use rand::seq::IteratorRandom;
use rand_distr::{Distribution, Normal, StandardNormal};

/// Struct to encapsulate all the random functionality.
pub struct NdarrRand {}

impl NdarrRand {
    /// Generates a random N-dimensional array with values x~U(low, high)
    pub fn uniform<
        T: SampleUniform + Debug + Clone + Copy + Default,
        R: Unsigned,
        D: Into<Dim<R>>,
    >(
        low: T,
        high: T,
        shape: D,
    ) -> Ndarr<T, R> {
        let d: Dim<R> = shape.into();
        let n = d.get_number_elements();
        let mut rng = rand::thread_rng();
        let random_vector: Vec<T> = (0..n)
            .map(|_| rng.sample(Uniform::new(low, high)))
            .collect();
        Ndarr::new(&random_vector, d).unwrap()
    }
    pub fn choose<T: Debug + Clone + Default, R: Unsigned, D: Into<Dim<R>>>(
        elements: &[T],
        shape: D,
    ) -> Ndarr<T, R> {
        let d: Dim<R> = shape.into();
        let n = d.get_number_elements();
        let mut rng = rand::thread_rng();
        let elements = elements.iter();
        let random_vector: Vec<T> = (0..n)
            .map(|_| elements.clone().choose(&mut rng).unwrap().clone())
            .collect();
        Ndarr::new(&random_vector, d).unwrap()
    }
    /// Generates a random N-dimensional array with values x~N(μ, σ^2)
    pub fn normal<T: Debug + Clone + Copy + Default + Float, R: Unsigned, D: Into<Dim<R>>>(
        mu: T,
        sigma: T,
        shape: D,
    ) -> Ndarr<T, R>
    where
        StandardNormal: Distribution<T>,
    {
        let d: Dim<R> = shape.into();
        let n = d.get_number_elements();
        let mut rng = rand::thread_rng();
        let normal = Normal::new(mu, sigma).unwrap();
        let random_vector: Vec<T> = (0..n).map(|_| normal.sample(&mut rng)).collect();
        Ndarr::new(&random_vector, d).unwrap()
    }
}

#[cfg(test)]
mod rng_arr {
    //use super::*;
    #[test]
    fn test_rand() {
        //let a = NdarrRand::uniform(0.0, 1.0, [2,2]);
    }
}
