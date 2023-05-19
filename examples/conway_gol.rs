use rapl::utils::random::NdarrRand;
use rapl::*;
use std::{
    io::{stdout, Write},
    thread::sleep,
    time::Duration,
};
const N: usize = 17;
const STEPS: usize = 100;

fn update(mat: &mut Ndarr<i8,U2>){
    let dirs = Ndarr::from([1,0,-1]);
    let out = dirs.map(|r| mat.roll(*r, 0)).outer_product(&dirs, |a, r| a.roll(r, 1)).sum();
    mat.bimap_in_place(&out, |prev, new| if new == 3|| (prev == 1 &&  (new == 4)){1}else{0})
}

fn main(){
    let mut x = NdarrRand::choose(&[0,1], [N,N]);//initialize game matrix with random 0 or 1
    let mut stdout = stdout();
    stdout.flush().unwrap();
    stdout.write_all(b"\x1B[2J\x1B[1;1H").unwrap();

    for _ in 0..STEPS{
        update(&mut x); //call update function
        let vis = x.map(|x| {
            if *x == 0 {
                "░".to_string()
            } else {
                "█".to_string()
            }
        }); //make it pretty
        println!("{}",vis);
        sleep(Duration::from_millis(40));
        stdout.write_all(b"\x1B[1;1H").unwrap();
        stdout.flush().unwrap();
    }
   stdout.write_all(b"\x1B[2J\x1B[1;1H").unwrap();
}