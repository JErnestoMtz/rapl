//! Conway's Game of Life on a wrapping grid, rendered in the terminal.

use rapl::*;
use std::{
    io::{stdout, Write},
    thread::sleep,
    time::Duration,
};
const N: usize = 17;
const STEPS: usize = 100;

fn update(mat: &mut Ndarr<i8, U2>) {
    let rolls = Ndarr::from([1, 0, -1]);
    let out = rolls
        .map(|r| mat.roll(*r, 0))
        .outer_product(&rolls, |a, r| a.roll(*r, 1))
        .unwrap()
        .iter_elems()
        .cloned()
        .reduce(|acc, shifted| acc + shifted)
        .expect("the neighborhood is non-empty");
    mat.zip_with_in_place(&out, |prev, new| {
        if *new == 3 || (*prev == 1 && (*new == 4)) {
            1
        } else {
            0
        }
    })
    .expect("the neighbor count has the grid's shape")
}

fn main() {
    let mut state = 0xC0FFEE_u64;
    let mut x = Ndarr::from_fn([N, N], |_| {
        state = state.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
        ((state >> 33) & 1) as i8
    });
    let mut stdout = stdout();
    stdout.flush().unwrap();
    stdout.write_all(b"\x1B[2J\x1B[1;1H").unwrap();

    for i in 0..STEPS {
        update(&mut x);
        let vis = x.map(|x| {
            if *x == 0 {
                "░".to_string()
            } else {
                "█".to_string()
            }
        });
        println!("{}", vis);
        println!(
            "\n Conway's Game of Life using rapl: [Step {} out of {}] \n",
            i + 1,
            STEPS
        );
        sleep(Duration::from_millis(50));
        stdout.write_all(b"\x1B[1;1H").unwrap();
        stdout.flush().unwrap();
    }
    stdout.write_all(b"\x1B[2J\x1B[1;1H").unwrap();
}
