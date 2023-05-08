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



pub(crate) fn format_vla(val: String, size: &usize) -> String {
    let mut s = val.clone();
    let l = val.len();
    s += ",";
    s += &" ".repeat(size - l);
    s
}


