use num_traits::Float;

use super::*;

impl<T: Float + Default + Clone + Debug, const R: usize> Ndarr<T, R> {
    //Threshold
    pub fn threshold(&self, threshold: &T, value: &T) -> Self {
        self.map(|x| if x > threshold { *x } else { *value })
    }

    //Hard Tanh
    pub fn hard_tanh(&self, min_val: &T, max_val: &T) -> Self {
        self.map(|x| {
            if x > max_val {
                *max_val
            } else if x < min_val {
                *min_val
            } else {
                *x
            }
        })
    }

    //Expontential Linear Unit function
    pub fn elu(&self, alpha: &T) -> Self {
        self.map(|x| {
            if x > &T::zero() {
                *x
            } else {
                *alpha * (x.exp() - T::one())
            }
        })
    }

    //Hard Shrinkage function
    pub fn hard_shrink(&self, lambda: &T) -> Self {
        self.map(|x| {
            if x > lambda || x < &-*lambda {
                //TODO: Is there a better way writing -lambda?
                -*x
            } else {
                T::zero()
            }
        })
    }

    //Hardsigmoid function
    pub fn hard_sigmoid(&self) -> Self {
        //TODO: Is there a better way to express 2, 3 and 6?
        let two = T::one() + T::one();
        let three = T::one() + T::one() + T::one() + T::one() + T::one() + T::one();
        let six = T::one() + T::one() + T::one() + T::one() + T::one() + T::one();
        self.map(|x| {
            if x <= &-three {
                T::zero()
            } else if x >= &three {
                T::one()
            } else {
                *x / six + T::one() / two
            }
        })
    }

    //Hardswish function
    pub fn hard_swish(&self) -> Self {
        //TODO: Is there a better way to express 3 and 6?
        let three = T::one() + T::one() + T::one() + T::one() + T::one() + T::one();
        let six = T::one() + T::one() + T::one() + T::one() + T::one() + T::one();
        self.map(|x| {
            if x <= &-three {
                T::zero()
            } else if x >= &three {
                *x
            } else {
                *x * (*x + three) / six
            }
        })
    }

    //LogSigmoid function ln( 1/(1+exp(-x)) )
    pub fn log_sigmoid(&self) -> Self {
        self.map(|x| (T::one() / (T::one() + (-*x).exp())).ln())
    }

    //Relu6 function min(max(0,x),6)
    pub fn relu_6(&self) -> Self {
        let six = T::one() + T::one() + T::one() + T::one() + T::one() + T::one();
        self.map(|x| x.max(T::zero()).min(six))
    }

    //Selu
    pub fn selu(&self) -> Self {
        let alpha = T::from(1.6732632423543772848170429916717).unwrap();
        let scale = T::from(1.6732632423543772848170429916717).unwrap();
        self.map(|x| scale * (x.max(T::zero())) + T::zero().min(alpha * (x.exp() - T::one())))
    }

    //Celu function max(0,x) + min(0, alpha * (exp(x/alpha)-1))
    pub fn celu(&self, alpha: &T) -> Self {
        self.map(|x| x.max(T::zero()) + T::zero().min(*alpha * ((*x / *alpha).exp() - T::one())))
    }

    //Silu function x*sigmoid(x)
    pub fn silu(&self) -> Self {
        self.map(|x| *x * T::one() / (T::one() + (-*x).exp()))
    }

    //Softplus function 1/beta * log(1 + exp(beta*x))
    pub fn softplus(&self, beta: &T) -> Self {
        self.map(|x| T::one() / *beta * (T::one() + (*beta * *x).exp()).ln())
    }

    //Mish function x * Tanh(Softplus(x, beta=1))
    pub fn mish(&self) -> Self {
        self.map(|x| *x * (T::one() + x.exp()).ln().tanh())
    }

    //Soft shrinkage function
    pub fn softshrink(&self, lambda: &T) -> Self {
        self.map(|x| {
            if x > lambda {
                *x - *lambda
            } else if x < &-*lambda {
                *x + *lambda
            } else {
                T::zero()
            }
        })
    }

    //Softsign function x/(1+abs(x))
    pub fn softsign(&self) -> Self {
        self.map(|x| *x / (T::one() + x.abs()))
    }

    //Tanhshrink function x-tanh(x)
    pub fn tanhshrink(&self) -> Self {
        self.map(|x| *x - x.tanh())
    }

    //Sigmoid function
    pub fn sigmoid(&self) -> Self {
        self.map(|x| T::one() / (T::one() + (-*x).exp()))
    }

    //Relu
    pub fn relu(&self) -> Self {
        self.map(|x| x.max(T::zero()))
    }

    //LeakyRelu
    pub fn leaky_relu(&self, a: T) -> Self {
        self.map(|x| x.max(a * *x))
    }

    //Softmax
    pub fn softmax(&self) -> Self {
        let max = self.maxf();
        let exp = self.map(|x| *x - max).exp();
        let sum = exp.sum();
        exp.map(|x| *x / sum)
    }
}

#[cfg(test)]
mod test_act {
    use super::*;

    #[test]
    fn threshold() {
        let x = Ndarr::from([-1., 0., 1., 2., 3.]);
        assert_eq!(
            x.threshold(&1.0, &42.0),
            Ndarr::from([42., 42., 42., 2., 3.])
        );
    }

    #[test]
    fn hard_tanh() {
        let x = Ndarr::from([-2., -1., 0., 1., 2., 3.]);
        assert_eq!(
            x.hard_tanh(&-1.5, &2.0),
            Ndarr::from([-1.5, -1., 0., 1., 2., 2.])
        );
    }

    #[test]
    fn elu() {
        let x = Ndarr::from([-2., -1., 0., 1., 2., 3.]);
        assert!(x.elu(&1.5).approx(&Ndarr::from([
            -1.29699707514508096215900075,
            -0.9481808382428365176067,
            0.,
            1.,
            2.,
            3.
        ])));
    }

    #[test]
    fn hard_shrink() {}

    #[test]
    fn hard_sigmoid() {}

    #[test]
    fn hard_swish() {}

    #[test]
    fn log_sigmoid() {}

    #[test]
    fn relu_6() {}

    #[test]
    fn selu() {}

    #[test]
    fn celu() {}

    #[test]
    fn silu() {}

    #[test]
    fn softplus() {}

    #[test]
    fn mish() {}

    #[test]
    fn softshrink() {}

    #[test]
    fn sigmoid() {
        let x = Ndarr::from([0., 1., 2., 3., 4., 5.]);
        println!("{}", x.sigmoid()) //[0.5        0.73105858 0.88079708 0.95257413 0.98201379 0.99330715]
    }

    #[test]
    fn relu() {
        let x = Ndarr::from([-2., -1., 0., 1., 2.]);
        assert_eq!(x.relu(), Ndarr::from([0., 0., 0., 1., 2.]))
    }

    #[test]
    fn leaky_relu() {
        let x = Ndarr::from([-2., -1., 0., 1., 2.]);
        assert_eq!(x.leaky_relu(0.1), Ndarr::from([-0.2, -0.1, 0., 1., 2.]))
    }

    #[test]

    fn softmax() {
        let x = Ndarr::from([1., 2., 3.]);
        assert_eq!(
            x.softmax(),
            Ndarr::from([0.09003057317038046, 0.24472847105479764, 0.6652409557748218])
        );
    }
}
