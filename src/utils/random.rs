use super::*;
use rand::prelude::*;
use rand;
use rand::distributions::Uniform;
use rand::distributions::uniform::SampleUniform;
use rand::seq::IteratorRandom;

/// Struct to encapsulate all the random functionality.
pub struct NdarrRand{}

impl NdarrRand {
    ///Generates a random N-dimensional array with values uniformly distributed from 0.0 to 1.0
    pub fn rand_f32<R: Unsigned, D: Into<Dim<R>>>(shape: D) -> Ndarr<f32,R>{
        let d: Dim<R> = shape.into();
        let n = d.get_number_elements();
        let mut rng = rand::thread_rng();
        let random_vector: Vec<f32> = (0..n)
        .map(|_| rng.sample(Uniform::new(0.0, 1.0)))
        .collect();
        Ndarr::new(&random_vector, d).unwrap()
    }

    pub fn rand_f64<R: Unsigned, D: Into<Dim<R>>>(shape: D) -> Ndarr<f64,R>{
        let d: Dim<R> = shape.into();
        let n = d.get_number_elements();
        let mut rng = rand::thread_rng();
        let random_vector: Vec<f64> = (0..n)
        .map(|_| rng.sample(Uniform::new(0.0, 1.0)))
        .collect();
        Ndarr::new(&random_vector, d).unwrap()
    }
    pub fn uniform<T: SampleUniform + Debug + Clone + Default, R: Unsigned, D: Into<Dim<R>>>(range: (T,T), shape: D) -> Ndarr<T,R>{
        let d: Dim<R> = shape.into();
        let n = d.get_number_elements();
        let mut rng = rand::thread_rng();
        let random_vector: Vec<T> = (0..n)
        .map(|_| rng.sample(Uniform::new(range.0.clone(), range.1.clone())))
        .collect();
        Ndarr::new(&random_vector, d).unwrap()
    }
    pub fn choose<T: Debug + Clone + Default, R: Unsigned, D: Into<Dim<R>>>(elements: &[T], shape: D)-> Ndarr<T,R>{
        let d: Dim<R> = shape.into();
        let n = d.get_number_elements();
        let mut rng = rand::thread_rng();
        let elements = elements.iter();
        let random_vector: Vec<T> = (0..n)
        .map(|_| elements.clone().choose(&mut rng).unwrap().clone())
        .collect();
        Ndarr::new(&random_vector, d).unwrap()
    }

}


#[cfg(test)]
mod rng_arr{
    use super::*;
    #[test]
    fn test_rand(){
        //let x: Ndarr<f32, U2> = NdarrRand::rand_f32(&[4,4]);
        //let y: Ndarr<i32,U1> = NdarrRand::uniform((-10,10), &[10]);
        //let faces = NdarrRand::choose(&["😀","😎","😐","😢"], &[4,4]);

        //println!("{}", x);
        //println!("{}", y);
        //println!("{}", faces);
    }
}

