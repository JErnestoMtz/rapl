use std::ops;

pub(crate) fn muliply_list<T>(list: &[T], init: T)->T
where T: ops::MulAssign  + Copy
{
    let mut result: T = init;
    for x in list{
        result *= *x
    }
    result
}

// Given and index n of a flat array, calulate the indexes of such element in an N rank array of shape: shape
pub(crate) fn  get_indexes<const N: usize>(n: &usize, shape: &[usize; N])->[usize; N]
    where [usize; N]: Default,
{
    let mut ind: [usize; N] = Default::default();
    for i in (0..N).rev(){
        let carry = muliply_list(&shape[i+1..],1 as usize);
        ind[i] = ((n -(n % carry))  / carry) % shape[i]

    }
    return ind;
}

// given the indexes of an element in a N rank array of shape shape return the position of such elemnt if the array was flattened
pub(crate) fn get_flat_pos<const N: usize>(indexes: &[usize; N], shape: &[usize; N])->Result<usize,String>{
    let mut ind: usize = 0;
    for i in 0..N{
        if indexes[i] >= shape[i]{
            return Err("Error: Index out of bounds".into());
        }
        ind += indexes[N-i-1]*muliply_list(&shape[N-i..], 1);
    }
    Ok(ind)
}