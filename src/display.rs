use super::*;
impl<T: Clone + Debug + Default + Display, R: Unsigned> Display for Ndarr<T, R> {
    // Kind of nasty function, it can be imprube a lot, but I think there is no scape from recursion.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let max_len = self.data.iter().map(|x| x.to_string().len()).max().unwrap();
        format_array(self.clone(), f, 0, self.rank(),max_len)
    }
}
fn format_vec(f: &mut fmt::Formatter<'_>, length: usize, limit: usize, separator: &str, ellipsis: &str,fmt_elem: &mut dyn FnMut(&mut fmt::Formatter, usize) -> fmt::Result)->fmt::Result{
    if length == 0{
    }else if length <= limit{
        fmt_elem(f, 0)?;
        for i in 1..length{
            f.write_str(separator)?;
            fmt_elem(f,i)?;
        }
    }else{
        let edge = limit / 2;
        fmt_elem(f,0)?;
        for i in 1..edge{
            f.write_str(separator)?;
            fmt_elem(f,i)?;
        }
        f.write_str(separator)?;
        f.write_str(ellipsis)?;
        for i in length - edge..length{
            f.write_str(separator)?;
            fmt_elem(f,i)?;
        }
    }
    Ok(())
}

use std::{fmt, io::repeat};
fn format_array<T,R: Unsigned>(arr: Ndarr<T,R>, f: &mut fmt::Formatter<'_>, dim: usize, full_dim: usize, max_len: usize)->fmt::Result
where T: Display + Clone + Default + Debug,
{
    match arr.shape() {
        &[] => f.write_str(&arr.data[0].to_string())?,
        &[len] => {
            f.write_str("[")?;
            format_vec(f, len, 100, ", ", "...", &mut |f, index| {
                let elm = arr.data[index].to_string();
                let path = max_len - elm.len();
                let elm = " ".repeat(path) + &elm;
                f.write_str(&elm)    
            })?;
            f.write_str("]")?;
        }
        shape => {
            let nl = "\n".repeat(shape.len()- 2);
            let indent=  " ".repeat(dim +1);
            let separator = format!(",\n{}{}", nl, indent);
            f.write_str("[")?;
            let limit = 100;
            format_vec(f, shape[0], limit, &separator, "...", &mut |f, index| {
                format_array(arr.slice_at_notyped(0)[index].clone(), f, dim + 1, full_dim, max_len)
            })?;
            f.write_str("]")?;
        }

    }
    Ok(())
}


#[cfg(test)]

mod disp{
    use super::*;
    #[test]
    fn disp_test(){
        let a = Ndarr::from(80..80+64).reshape([4,4,4]).unwrap();
        println!("a = \n {}", a);
 //       a = 
 //[[[ 80,  81,  82,  83],
  //[ 84,  85,  86,  87],
  //[ 88,  89,  90,  91],
  //[ 92,  93,  94,  95]],

 //[[ 96,  97,  98,  99],
  //[100, 101, 102, 103],
  //[104, 105, 106, 107],
  //[108, 109, 110, 111]],

 //[[112, 113, 114, 115],
  //[116, 117, 118, 119],
  //[120, 121, 122, 123],
  //[124, 125, 126, 127]],

 //[[128, 129, 130, 131],
  //[132, 133, 134, 135],
  //[136, 137, 138, 139],
  //[140, 141, 142, 143]]]
    }
}