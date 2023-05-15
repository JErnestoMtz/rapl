use rapl::utils::random::NdarrRand;
use rapl::*;
use std::{
    io::{stdout, Write},
    thread::sleep,
    time::Duration,
};

const N: usize = 18; // Number of spins in each direction.
const T: f32 = 0.05; // Temperature
const STEPS: usize = 700; //Steps of simulation
const M: usize = 20; //Number of updates per metropolis circle


fn metropolis(spin_arr: &mut Ndarr<f32, U2>){  
    let energy:Ndarr<f32, U2> = 2. * spin_arr.clone() * (spin_arr.roll(1, 0)+
                                                    spin_arr.roll(-1, 0) +
                                                    spin_arr.roll(1, 1) +
                                                    spin_arr.roll(-1, 1));
    let temp_exp = (- &energy / T).exp(); 
    let indexes: Vec<usize> = (0..N).collect();
    let i_s = NdarrRand::choose(&indexes, [M]); //random i indexes
    let j_s = NdarrRand::choose(&indexes, [M]); //random j indexes
    let p_swich: f32 = NdarrRand::rand_f32([1])[0]; // selection probability of random spin switch
    for (i, j) in i_s.data.iter().zip(j_s.data.iter()) {
        if energy[[*i, *j]] < 0.0 || p_swich < temp_exp[[*i, *j]] {
            spin_arr[[*i, *j]] *= &-1.0;
        }
    }
}

fn main() {
    let mut stdout = stdout();
    let mut spin_arr = NdarrRand::choose(&[-1.0, 1.0], [N, N]);
    stdout.flush().unwrap();
    stdout.write_all(b"\x1B[2J\x1B[1;1H").unwrap();
    for i in 0..STEPS {
        metropolis(&mut spin_arr); //updates array with metropolis algorithm
        let vis = spin_arr.map_types(|x| {
            if *x < 0.0 {
                "░".to_string()
            } else {
                "█".to_string()
            }
        }); //make it pretty

        println!("{}", vis);

        println!(
            "\n The Ising Model using rapl: [Step {} out of {}] \n \n for more info: https://en.wikipedia.org/wiki/Ising_model"
        , i + 1, STEPS);
        sleep(Duration::from_millis(5));
        stdout.write_all(b"\x1B[1;1H").unwrap();
        stdout.flush().unwrap();
    }
    stdout.write_all(b"\x1B[2J\x1B[1;1H").unwrap();
    stdout.flush().unwrap();
}
