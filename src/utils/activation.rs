use num_traits::Float;

use super::*;

impl<T: Float + Default + Clone + Debug, R: Unsigned> Ndarr<T, R> {
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
                *x
            } else {
                T::zero()
            }
        })
    }

    //Hardsigmoid function
    pub fn hard_sigmoid(&self) -> Self {
        self.map(|x| {
            if x <= &T::from(-3.0).unwrap() {
                T::zero()
            } else if x >= &T::from(3.0).unwrap() {
                T::one()
            } else {
                *x / T::from(6.0).unwrap() + T::one() / T::from(2.0).unwrap()
            }
        })
    }

    //Hardswish function
    pub fn hard_swish(&self) -> Self {
        self.map(|x| {
            if x <= &T::from(-3.0).unwrap() {
                T::zero()
            } else if x >= &T::from(3.0).unwrap() {
                *x
            } else {
                *x * (*x + T::from(3.0).unwrap()) / T::from(6.0).unwrap()
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
        let scale = T::from(1.0507009873554804934193349852946).unwrap();
        self.map(|x| scale * (x.max(T::zero()) + T::zero().min(alpha * (x.exp() - T::one()))))
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

//#[cfg(test)]
//mod test_act {
    //use super::*;

    //#[test]
    //fn threshold() {
        //let x = Ndarr::from([-1., 0., 1., 2., 3.]);
        //assert_eq!(
            //x.threshold(&1.0, &42.0),
            //Ndarr::from([42., 42., 42., 2., 3.])
        //);
    //}

    //#[test]
    //fn hard_tanh() {
        //let x = Ndarr::from([-2., -1., 0., 1., 2., 3.]);
        //assert_eq!(
            //x.hard_tanh(&-1.5, &2.0),
            //Ndarr::from([-1.5, -1., 0., 1., 2., 2.])
        //);
    //}

    //#[test]
    //fn elu() {
        //let x = Ndarr::from([-2., -1., 0., 1., 2., 3.]);
        //assert!(x.elu(&1.5).approx(&Ndarr::from([
            //-1.29699707514508096215900075,
            //-0.9481808382428365176067,
            //0.,
            //1.,
            //2.,
            //3.
        //])));
    //}

    //#[test]
    //fn hard_shrink() {
        //let x = Ndarr::from([-2., -1., 0., 1., 2., 3.]);
        //assert_eq!(
            //x.hard_shrink(&1.0,),
            //Ndarr::from([-2., -0., 0., 0., 2., 3.])
        //);
    //}

    //#[test]
    //fn hard_sigmoid() {
        //let x = Ndarr::from([-3., -2., -1., 0., 1., 2., 3.]);
        //assert!(x.hard_sigmoid().approx(&Ndarr::from([
            //0.,
            //0.1666666666666666666666666666,
            //0.3333333333333333333333333333,
            //0.5,
            //0.6666666666666666666666666666,
            //0.8333333333333333333333333333,
            //1.
        //])));
    //}

    //#[test]
    //fn hard_swish() {
        //let x = Ndarr::from([-3., -2., -1., 0., 1., 2., 3.]);
        //assert!(x.hard_swish().approx(&Ndarr::from([
            //0.,
            //-0.3333333333333333333333333333,
            //-0.3333333333333333333333333333,
            //0.,
            //0.66666666666666666666666666666,
            //1.66666666666666666666666666666,
            //3.
        //])));
    //}

    //#[test]
    //fn log_sigmoid() {
        //let x = Ndarr::from([-3., -2., -1., 0., 1., 2., 3.]);
        //assert!(x.log_sigmoid().approx(&Ndarr::from([
            //-3.0485873515737420587589259198,
            //-2.1269280110429724964437268063,
            //-1.3132616875182228340489954949,
            //-0.6931471805599453094172321214,
            //-0.3132616875182228340489954949,
            //-0.1269280110429724964437268063,
            //-0.0485873515737420587589259198
        //])));
    //}

    //#[test]
    //fn relu_6() {
        //let x = Ndarr::from([-3., -2., -1., 0., 1., 2., 3., 4., 5., 6., 7.]);
        //assert_eq!(
            //x.relu_6(),
            //Ndarr::from([0., 0., 0., 0., 1., 2., 3., 4., 5., 6., 6.])
        //);
    //}

    //#[test]
    //fn selu() {
        //let x = Ndarr::from([-3., -2., -1., 0., 1., 2., 3.]);
        //assert!(x.selu().approx(&Ndarr::from([
            //-1.6705687287671119749192485076,
            //-1.5201664685956950351375928376,
            //-1.1113307378125627617986406624,
            //0.,
            //1.05070098735548049341933498529,
            //2.10140197471096098683866997058,
            //3.15210296206644148025800495588
        //])));
    //}

    //#[test]
    //fn celu() {
        //let x = Ndarr::from([-3., -2., -1., 0., 1., 2., 3.]);
        //assert!(x.celu(&1.5).approx(&Ndarr::from([
            //-1.2969970751450809621590007575,
            //-1.1046042928264098448814490815,
            //-0.7298743214511119596920203608,
            //0.,
            //1.,
            //2.,
            //3.
        //])));
    //}

    //#[test]
    //fn silu() {
        //let x = Ndarr::from([-3., -2., -1., 0., 1., 2., 3.]);
        //assert!(x.silu().approx(&Ndarr::from([
            //-0.1422776195327003426365444553,
            //-0.2384058440442351118805417173,
            //-0.2689414213699951207488407581,
            //0.,
            //0.73105857863000487925115924182,
            //1.76159415595576488811945828260,
            //2.85772238046729965736345554468
        //])));
    //}

    //#[test]
    //fn softplus() {
        //let x = Ndarr::from([-3., -2., -1., 0., 1., 2., 3.]);
        //assert!(x.softplus(&1.5).approx(&Ndarr::from([
            //0.00736516323239587754776281247,
            //0.03239156771582803917261727990,
            //0.13427551865516827299965520603,
            //0.46209812037329687294482141430,
            //1.13427551865516827299965520603,
            //2.03239156771582803917261727990,
            //3.00736516323239587754776281247
        //])));
    //}

    //#[test]
    //fn mish() {
        //let x = Ndarr::from([-3., -2., -1., 0., 1., 2., 3.]);
        //assert!(x.mish().approx(&Ndarr::from([
            //-0.1456474612756245873146171857,
            //-0.2525014826957088636350729038,
            //-0.3034014613741089180743892753,
            //0.,
            //0.86509838826731034611623344925,
            //1.94395895953399452031848132479,
            //2.98653500496795731905705182010
        //])));
    //}

    //#[test]
    //fn softshrink() {
        //let x = Ndarr::from([-3., -2., -1., 0., 1., 2., 3.]);
        //assert_eq!(
            //x.softshrink(&1.5),
            //Ndarr::from([-1.5, -0.5, 0., 0., 0., 0.5, 1.5])
        //);
    //}

    //#[test]
    //fn sigmoid() {
        //let x = Ndarr::from([0., 1., 2., 3., 4., 5.]);
        //assert!(x.sigmoid().approx(&Ndarr::from([
            //0.5,
            //0.73105857863000487925115924182,
            //0.88079707797788244405972914130,
            //0.95257412682243321912115184822,
            //0.98201379003790844197320686205,
            //0.99330714907571514444063801961
        //])));
    //}

    //#[test]
    //fn relu() {
        //let x = Ndarr::from([-2., -1., 0., 1., 2.]);
        //assert_eq!(x.relu(), Ndarr::from([0., 0., 0., 1., 2.]))
    //}

    //#[test]
    //fn leaky_relu() {
        //let x = Ndarr::from([-2., -1., 0., 1., 2.]);
        //assert_eq!(x.leaky_relu(0.1), Ndarr::from([-0.2, -0.1, 0., 1., 2.]))
    //}

    //#[test]

    //fn softmax() {
        //let x = Ndarr::from([1., 2., 3.]);
        //assert!(x.softmax().approx(&Ndarr::from([
            //0.09003057317038046,
            //0.24472847105479764,
            //0.6652409557748218
        //])));
    //}
//}
