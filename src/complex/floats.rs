use super::*;
use num_traits::Float;

impl<T: Float> C<T> {
    pub fn abs(&self) -> T {
        self.0.hypot(self.1)
    }
    pub fn arg(&self) -> T {
        self.1.atan2(self.0)
    }
    pub fn exp(&self) -> Self {
        C(self.0.exp() * self.1.cos(), self.0.exp() * self.1.sin())
    }
    pub fn sqrt(&self) -> Self {
        // Branch cut: -pi/2 <= arg(sqrt(z)) <= pi/2.
        let two = T::one() + T::one();
        let (r, arg) = self.to_polar();
        C(r.sqrt(), T::zero()) * C(T::zero(), arg / two).exp()
    }
    pub fn ln(&self) -> Self {
        // Main branch complex logarithm, ln(-1) = iπ (as in numpy).
        C(self.abs().ln(), self.arg())
    }
    pub fn powf(&self, n: T) -> Self {
        let (r, arg) = self.to_polar();
        Self::from_polar(r.powf(n), arg * n)
    }
    pub fn powc(&self, c: Self) -> Self {
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

    pub fn from_polar(r: T, angle: T) -> C<T> {
        let unit = C(T::from(0).unwrap(), angle).exp();
        C(r * unit.0, r * unit.1)
    }

    pub fn to_polar(&self) -> (T, T) {
        (self.abs(), self.arg())
    }

    pub fn is_infinite(&self) -> bool {
        self.0.is_infinite() || self.1.is_infinite()
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
