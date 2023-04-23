use num_traits::Float;

use super::*;

impl<T: Float + Default + Clone + Debug, const R: usize> Ndarr<T, R> {
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
