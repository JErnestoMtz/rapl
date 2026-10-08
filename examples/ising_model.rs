//! 2-D Ising model via the Metropolis algorithm, rendered in the terminal.

use rapl::*;
use std::{
    io::{stdout, Write},
    thread::sleep,
    time::Duration,
};

const N: usize = 18; // Spins per direction.
const T: f32 = 0.05; // Temperature
const STEPS: usize = 700; // Simulation steps
const M: usize = 20; // Updates per Metropolis sweep

/// SplitMix64: every output bit is mixed, so `& 1` and `% N` stay uniform.
fn next_u64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn metropolis(spin_arr: &mut Ndarr<f32, U2>, rng: &mut u64) {
    let energy: Ndarr<f32, U2> = 2.
        * spin_arr.clone()
        * (spin_arr.roll(1, 0) + spin_arr.roll(-1, 0) + spin_arr.roll(1, 1) + spin_arr.roll(-1, 1));
    let temp_exp = (-&energy / T).exp();
    let i_s = Ndarr::from_fn([M], |_| (next_u64(rng) as usize) % N);
    let j_s = Ndarr::from_fn([M], |_| (next_u64(rng) as usize) % N);
    let p_switch = (next_u64(rng) >> 40) as f32 / ((1u32 << 24) as f32);
    for (i, j) in i_s.data().iter().zip(j_s.data().iter()) {
        if energy[[*i, *j]] < 0.0 || p_switch < temp_exp[[*i, *j]] {
            spin_arr[[*i, *j]] *= -1.0;
        }
    }
}

fn main() {
    let mut stdout = stdout();
    let mut rng = 0xC0FFEE_u64;
    let mut spin_arr = Ndarr::from_fn([N, N], |_| {
        if next_u64(&mut rng) & 1 == 0 {
            -1.0
        } else {
            1.0
        }
    });
    stdout.flush().unwrap();
    stdout.write_all(b"\x1B[2J\x1B[1;1H").unwrap();
    for i in 0..STEPS {
        metropolis(&mut spin_arr, &mut rng);
        let vis = spin_arr.map(|x| {
            if *x < 0.0 {
                "░".to_string()
            } else {
                "█".to_string()
            }
        });

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
