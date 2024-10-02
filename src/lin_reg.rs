use ndarray::{Array1, Array2, Axis};
use rand::{thread_rng, Rng, seq::SliceRandom};

pub struct LinearRegressionParameters {
    pub learning_rate: f64,
    pub epochs: usize,
    pub batch_size: usize

}

impl Default for LinearRegressionParameters {
    fn default() -> Self {
        LinearRegressionParameters::new()
    }
}

impl LinearRegressionParameters {
    pub fn new() -> Self {
        LinearRegressionParameters {
            learning_rate: 0.01,
            epochs: 100,
            batch_size: 0
        }
    }

    pub fn learning_rate(mut self, learning_rate: f64) -> Self {
        if learning_rate <= 0.0 {
            panic!{"Learning rate must be greater than zero"};
        }
        self.learning_rate = learning_rate;
        self
    }
    
    pub fn epochs(mut self, epochs: usize) -> Self {
        if epochs == 0 {
            panic!("Number of epochs must be greater than zero");
        }
        self.epochs = epochs;
        self
    }

    pub fn batch_size(mut self, batch_size: usize) -> Self {
        if batch_size == 0 {
            println!("If batch size is 0, then it will be equal to the number of samples");
        }
        self.batch_size = batch_size;
        self
    }
}

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

    pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>, params: LinearRegressionParameters) {
        let (n_samples, n_features) = x.dim();
        
        assert_eq!(y.len(), n_samples);

        let mut batch_indexes: Vec<usize> = (0..n_samples).collect();

        if params.batch_size > n_samples {
            println!("#[WARNING] Batch size should be less than number of samples. It will be equal to the number of samples: {}", &n_samples);
        } else {
            batch_indexes.shuffle(&mut thread_rng());
            batch_indexes = batch_indexes[0..params.batch_size].to_vec();
        }

        for _ in 0..params.epochs {
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
    
            let mut intercept = match &self.intercept {
                0. => 0.,
                _ => self.intercept
            };

            for &i in &batch_indexes {
                let x_row = match x.select(Axis(0), &[i]).into_shape_clone((x.dim().1, )) {
                    Ok(row) => row,
                    Err(_) => panic!("Failed to get a string by index {i} from the array")
                };

                weights -= &((2.0 / params.batch_size as f64) * &x_row * (y[i] - (&x_row.dot(&weights) + intercept)));
                intercept -= (2.0 / params.batch_size as f64) * (y[i] - (&x_row.dot(&weights) + intercept));
            }

            self.weights = Some(weights);
            self.intercept = intercept;
        }
    }

    pub fn predict(&self, x: &Array2<f64>) -> Array1<f64> {
        match &self.weights {
            None => panic!("The linear regression has to be fitted!"),
            Some(weights) => x.dot(weights) + self.intercept
        }
    }
}