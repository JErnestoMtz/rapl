# rapl

*NOTE*: `rapl`  requires Nightly and is strictly intended for non-production purposes only. `rapl` utilizes certain unstable features that may result in unexpected behavior, and is not optimized for performance.

`rapl` is an experimental numerical computing Rust create for working with N-dimensional array, along with a wide range of mathematical functions to manipulate them. It takes inspiration from NumPy and APL, with the primary aim of achieving maximum ergonomic and user-friendliness while maintaining generality. Notably, it offers automatic Rank Polymorphic broadcasting between arrays of varying shapes and scalars as a built-in feature.

```Rust
#![feature(generic_const_exprs)]
use rapl::*;
fn main() {
    let a = Ndarr::from([1, 2, 3]);
    let b = Ndarr::from([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
    let r = a + b - 1;
    assert_eq!(r, Ndarr::from([[1, 3, 5], [4, 6, 8], [7, 9, 11]]));
}
```

### Array initialization
There are multiple handy ways of initializing N-dimensional arrays (or `Ndarr`).
- From Native Rust arrays to `Ndarr`.
```Rust 
let a = Ndarr::from(["a","b","c"]); 
let b = Ndarr::from([[1,2],[3,4]]);
```
- From ranges.
```Rust
let a = Ndarr::from(1..7).reshape(&[2,3])
```
- From `&str`
```Rust
let chars = Ndarr::from("Hello rapl!"); //Ndarr<char,1>
```
- Others:
```Rust 
let ones: Ndarr<f32, 2> = Ndarr::ones(&[4,4]);
let zeros : Ndarr<i32, 3>= Ndarr::zeros(&[2,3,4]);
let letter_a = Ndarr::fill("a", &[5]);
let fold = Ndarr::new(data: &[0, 1, 2, 3], shape: [2, 2]).expect("Error initializing");
```

### Element wise operations
- Arithmetic operation with with scalars
```Rust
let ones: Ndarr<i32, 2> = Ndarr::ones(&[4,4]);
let twos = &ones + 1;
let sixes = &twos * 3;
```
- Arithmetic operation between `Ndarr`s,
```Rust
let a = Ndarr::from([[1,2],[3,4]]);
let b = Ndarr::from([[1,2],[-3,-4]]);

assert_eq!(a + b, Ndarr::from([[2,4],[0,0]]))
```
Note: If the shapes are not equal `rapl` will automatically broadcast the arrays into a compatible shape (if it exist) and perform the operation.
- Math operations including trigonometric functions
```Rust
let x = Ndarr::from([-1.0 , -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0]);
let sin_x = &x.sin();
let cos_x = &x.sin();
let tanh_x = &x.tanh();

let abs_x = x.abs();
```
- Map function
```Rust
let a = Ndarr::from([[1,2],[3,4]]);
let mapped = a.map(|x| x**2-1);
```
### Monadic tensor operations
- Transpose
```Rust
let arr = Ndarr::from([[1,2,3],[4,5,6]]);	
assert_eq!(arr.shape(), [2,3]);
assert_eq!(arr.clone().t().shape, [3,2]); //transpose
```
- Reshape
```Rust
let a = Ndarr::from(1..7).reshape(&[2,3]).unwrap();
```
- Slice
```Rust
let arr = Ndarr::from([[1,2],[3,4]]);

assert_eq!(arr.slice_at(1)[0], Ndarr::from([1,3]))
```
- Reduce
```Rust
let sum_axis = arr.clone().reduce(1, |x,y| x + y).unwrap();
assert_eq!(sum_axis, Ndarr::from([6, 15])); //sum reduction
```

### Diatic tensor operations
- Generalized matrix multiplication between compatible arrays
```Rust
use rapl::*
use rapl::ops::{mat_mul};
let a = Ndarr::from(1..7).reshape(&[2,3]).unwrap();
let b = Ndarr::from(1..7).reshape(&[3,2]).unwrap();
    
let matmul = mat_mul(a, b))
```
- [APL](https://en.wikipedia.org/wiki/APL_(programming_language)) inspired Inner Product.
```Rust
    let a = Ndarr::from(1..7).reshape(&[2,3]).unwrap();
    let b = Ndarr::from(1..7).reshape(&[3,2]).unwrap();
    
    let inner = rapl::ops::inner_product(|x,y| x*y, |x,y| x+y, a.clone(), b.clone());
    assert_eq!(inner, rapl::ops::mat_mul(a, b))

```
- Outer Product.

```Rust
    let suits = Ndarr::from(["♣","♠","♥","♦"]);
    let ranks = Ndarr::from(["2","3","4","5","6","7","8","9","10","J","Q","K","A"]);

    let add_str = |x: &str, y: &str| (x.to_owned() + y);

    let deck = ops::outer_product( add_str, ranks, suits).flatten(); //All cards in a deck
```

