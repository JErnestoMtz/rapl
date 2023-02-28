
use std::ops;
use std::cmp::max;
use core::marker::Destruct;

pub(crate) fn multiply_list<T>(list: &[T], init: T) -> T
where
    T: ops::MulAssign + Copy,
{
    let mut result: T = init;
    for x in list {
        result *= *x
    }
    result
}

// Given and index n of a flat array, calculate the indexes of such element in an N rank array of shape: shape
pub(crate) fn get_indexes<const N: usize>(n: &usize, shape: &[usize; N]) -> [usize; N]
where
    [usize; N]: Default,
{
    let mut ind: [usize; N] = Default::default();
    for i in (0..N).rev() {
        let carry = multiply_list(&shape[i + 1..], 1 as usize);
        ind[i] = ((n - (n % carry)) / carry) % shape[i]
    }
    return ind;
}

pub(crate) fn get_indexes2<'a>(n: &usize, shape: &'a [usize]) -> Vec<usize> {
    let mut ind = Vec::with_capacity(shape.len());
    let n = shape.len();
    for i in (0..n).rev() {
        let carry = multiply_list(&shape[i + 1..], 1 as usize);
        ind[i] = ((n - (n % carry)) / carry) % shape[i]
    }
    return ind;
}

// given the indexes of an element in a N rank array of shape shape return the position of such element if the array was flattened
pub(crate) fn get_flat_pos<const N: usize>(
    indexes: &[usize; N],
    shape: &[usize; N],
) -> Result<usize, String> {
    let mut ind: usize = 0;
    for i in 0..N {
        if indexes[i] >= shape[i] {
            return Err("Error: Index out of bounds".into());
        }
        ind += indexes[N - i - 1] * multiply_list(&shape[N - i..], 1);
    }
    Ok(ind)
}

pub(crate) fn get_flat_pos2(indexes: Vec<usize>, shape: &[usize]) -> Result<usize, String> {
    let mut ind: usize = 0;
    let n = shape.len();
    for i in 0..n {
        if indexes[i] >= shape[i] {
            return Err("Error: Index out of bounds".into());
        }
        ind += indexes[n - i - 1] * multiply_list(&shape[n - i..], 1);
    }
    Ok(ind)
}

pub(crate) fn format_vla(val: String, size: &usize) -> String {
    let mut s = val.clone();
    let l = val.len();
    s += &",";
    s += &" ".repeat(size - l);
    s
}

pub fn remove_element<T: Copy, const N: usize>(arr: [T; N], index: usize) -> [T;  N - 1 ] {
    assert!(index < N);
    let mut result = [arr[0];  N - 1 ];
    let mut j = 0;
    for i in 0..N {
        if i != index {
            result[j] = arr[i];
            j += 1;
        }
    }
    result
}

pub const fn const_max<T:  ~const Ord + ~const Destruct>(a: T, b: T)->T{
    max(a, b)
}

fn index_or(arr: &[usize], index: usize, or: usize)->usize{
    if index >= arr.len(){
        or
    }else {
        arr[index]
    }
}

pub fn broadcast_shape<const N: usize, const M: usize>(shape1: &[usize; N], shape2: &[usize; M])->Result<[usize; {const_max(N,M)}], String>
where [usize; {const_max(N, M)}]: Default
{
    let mut out_shape: [usize; {const_max(N, M)}] = Default::default();
    let mut sh1 = shape1.to_vec();
    let mut sh2 = shape2.to_vec();
    sh1.reverse();
    sh2.reverse();

    let l = max(N,M);
    for i in 0..l{
        let size1 = index_or(&sh1, i, 1);
        let size2 = index_or(&sh2, i, 1);
        if size1 != 1 && size2 != 1 && size1 != size2{
            return Err(String::from(format!("Error arrays with shape {:?} and {:?} can not be broadcasted", shape1, shape2)))
        }
        out_shape[l-i-1] = max(size1, size2)
    }
    return Ok(out_shape);

}