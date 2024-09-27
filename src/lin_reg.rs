use ndarray::{ArrayBase, Array1, Ix1, Ix2, Data};
use rand::{thread_rng, Rng};

use super::gradient_descent::gradient_descent;

pub struct LinearRegression {
    pub weights: Option<Array1<f64>>,
    pub intercept: f64,
}

impl Default for LinearRegression {
    fn default() -> Self {
        LinearRegression::new()
    }
}

impl LinearRegression {
    pub fn new() -> Self {
        LinearRegression {
            weights: None,
            intercept: 0.,
        }
    }

    pub fn fit<A, B>(&mut self, x: &ArrayBase<A, Ix2>, y: &ArrayBase<B, Ix1>, learning_rate: f64, epochs: usize) 
    where
        A: Data<Elem = f64>,
        B: Data<Elem = f64>
    {
        let (n_samples, n_features) = x.dim();
        
        assert_eq!(y.len(), n_samples);

        let mut weights = match &self.weights {
            Some(w) => w.clone(),
            None => {
                let mut w = vec![0.0; n_features];
                for i in &mut w {
                    *i = thread_rng().gen_range(-1.0..1.0);
                }
                Array1::from_vec(w)
            }
        };

        let mut intercept = 0.0;

        for _ in 0..epochs {
            let new_weights;
            let new_intercept;
            (new_weights, new_intercept) = gradient_descent(&mut weights, &mut intercept, x, y, learning_rate);
            weights = new_weights;
            intercept = new_intercept;
            
        }

        self.weights = Some(weights);
        self.intercept = intercept;
    }

    pub fn predict<A>(&self, x: &ArrayBase<A, Ix2>) -> Array1<f64>
    where
        A: Data<Elem = f64>
    {
        match &self.weights {
            None => panic!("The linear regression has to be fitted!"),
            Some(weights) => x.dot(weights) + self.intercept
        }
    }
}