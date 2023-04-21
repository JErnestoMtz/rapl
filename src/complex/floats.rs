use super::*;
use num_traits::{Float};

impl<T: Float> C<T> {
    pub fn abs(&self) -> T {
        //(self.0 * self.0 + self.1 * self.1).sqrt()
        self.0.hypot(self.1)
    }
    pub fn arg(&self) -> T {
        // theta
        //(self.1 / self.0).atan()
        self.1.atan2(self.0)
    }
    pub fn exp(&self) -> Self {
        C(self.0.exp() * self.1.cos(), self.0.exp() * self.1.sin())
    }
    pub fn sqrt(&self) -> Self {
        //z = a + bi -> z = re^{iθ}
        //sqrt(z) = z^{1/2}
        //sqrt(z) = sqrt(r)e^{iθ/2}
        //The square root of z generally has two solutions, this functions has a branch cut that satisfy -pi/2 <= arg(sqrt(z)) <= pi/2.
        let two = T::one() + T::one();
        let (r, arg) = self.to_polar();
        C(r.sqrt(), T::zero()) * C(T::zero(), arg / two).exp()
    }
    pub fn ln(&self) -> Self {
        // Main branch complex logarithm with ln(-1)=iπ (as in numpy)
        // Note this function is not continuous since the complex logarithm is only continuous
        // if no closed curve around 0 exists. To make the complex logarithm continuous,
        // typically, a curve from 0 to infinity is removed from the input domain.
        // Commonly this curve is the negative real axis. Then ln(-1) is no longer defined.
        C(self.abs().ln(), self.arg())
    }
    pub fn powf(&self, n: T) -> Self {
        //z^n = r^n*exp(n*i*phi) with z = r*(cos(phi) + i*sin(phi))
        if n.is_zero() {
            return C(T::one(), T::zero());
        } else if n < T::zero() {
            //it seems faster this way
            let (r, arg) = self.to_polar();
            let pow_r = r.powf(-n);
            let pow_c = C(T::zero(), arg.mul(n)).exp();
            return C(T::one(), T::zero()) / C(pow_r * pow_c.0, pow_r * pow_c.1);
        } else {
            let (r, arg) = self.to_polar();
            let pow_r = r.powf(n);
            let pow_c = C(T::zero(), arg.mul(n)).exp();
            return C(pow_r * pow_c.0, pow_r * pow_c.1);
        }
    }
    pub fn powc(&self, c: Self) -> Self {
        //z = r*exp(iθ) = e^{ln(r) + iθ}
        //Z = (c + id)
        //z^Z = (r*exp(iθ))^{c + id}
        // z^Z = e ^ {ln(r)(c + id) + iθ(c + id)}
        let (r, arg1) = self.to_polar();
        let p1 = C(r.ln() * c.0, r.ln() * c.1);
        let p2 = C(T::zero(), arg1) * c;
        (p1 + p2).exp()
    }
    pub fn sin(&self) -> Self {
        let nom = C(-self.1, self.0).exp() - C(self.1, -self.0).exp();
        let den = C(T::from(0).unwrap(), T::from(2).unwrap());
        nom / den
    }
    pub fn cos(&self) -> Self {
        let nom = C(-self.1, self.0).exp() + C(self.1, -self.0).exp();
        let den = C(T::from(2).unwrap(), T::from(0).unwrap());
        nom / den
    }
    pub fn tan(&self) -> Self {
        self.sin() / self.cos()
    }

    pub fn cot(&self) -> Self {
        self.cos() / self.sin()
    }

    pub fn sec(&self) -> Self {
        C(T::from(1).unwrap(), T::from(0).unwrap()) / self.cos()
    }

    pub fn csc(&self) -> Self {
        C(T::from(1).unwrap(), T::from(0).unwrap()) / self.sin()
    }

    pub fn sinh(&self) -> Self {
        let nom = self.exp() - (-self).exp();
        let den = C(T::from(2).unwrap(), T::from(0).unwrap());
        nom / den
    }

    pub fn cosh(&self) -> Self {
        let nom = self.exp() + (-self).exp();
        let den = C(T::from(2).unwrap(), T::from(0).unwrap());
        nom / den
    }

    pub fn tanh(&self) -> Self {
        self.sinh() / self.cosh()
    }

    pub fn coth(&self) -> Self {
        self.cosh() / self.sinh()
    }

    pub fn sech(&self) -> Self {
        C(T::from(1).unwrap(), T::from(0).unwrap()) / self.cosh()
    }

    pub fn csch(&self) -> Self {
        C(T::from(1).unwrap(), T::from(0).unwrap()) / self.sinh()
    }

    pub fn from_polar(r: T, angle: T) -> C<T> {
        let unit = C(T::from(0).unwrap(), angle).exp();
        C(r * unit.0, r * unit.1)
    }

    pub fn to_polar(&self) -> (T, T) {
        (self.abs(), self.arg())
    }

    pub fn is_infinite(&self) -> bool {
        self.0.is_finite() && self.1.is_finite()
    }

    pub fn is_finite(&self) -> bool {
        self.0.is_finite() && self.1.is_finite()
    }

    pub fn is_normal(&self) -> bool {
        self.0.is_normal() && self.1.is_normal()
    }

    pub fn is_nan(&self) -> bool {
        self.0.is_nan() || self.1.is_nan()
    }
}
