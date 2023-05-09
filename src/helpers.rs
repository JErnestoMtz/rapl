use std::ops;

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
