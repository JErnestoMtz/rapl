use rapl::*;
use rapl::utils::rapl_img::open_lumaf32;
use image::ImageFormat;

fn main(){
        //open image as lumaf32 (gray scale) where 0.0 is white 1.0 is black.
        let img = open_lumaf32("graphics/peppers.png").unwrap();

        //transform to complex and take the 2D FFT
        let fft = img.to_complex().fft2d();

        //initialize kernel for convolution
        let mut kernel:Ndarr<f32,_> = Ndarr::zeros(&fft.dim);
        let (m,n) = (img.shape()[0], img.shape()[1]);
        let mid_x = m /2;
        let mid_y = n /2;
        kernel[[mid_x,mid_y]] = 4.;
        kernel[[mid_x + 1,mid_y]] = -1.;
        kernel[[mid_x-1,mid_y]] = -1.;
        kernel[[mid_x,mid_y+1]] = -1.;
        kernel[[mid_x,mid_y-1]] = -1.;
        //FFT the kernell
        let kernell = kernel.to_complex().fft2d();
        //multiply the image FFT to the kernel fft to then do the inverse transform to ger the convolution of the
        //image and the kernel
        let out = (fft * kernell).ifft2().fftshif().re();
        //save output image
        out.save_as_luma("graphics/pepper_edges.png", ImageFormat::Png)
}


