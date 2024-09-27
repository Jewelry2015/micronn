use ndarray::{ArrayBase, Array1, Ix1, Ix2, Data, Axis};
use rand::seq::SliceRandom;

pub fn stochastic_gradient_descent<A, B>(weights: &mut Array1<f64>, intercept: &mut f64, x: &ArrayBase<A, Ix2>, y: &ArrayBase<B, Ix1>, learning_rate: f64, batch_size: usize) -> (Array1<f64>, f64)
where
    A: Data<Elem = f64>,
    B: Data<Elem = f64>
{
    let n = x.dim().0;

    let mut weights_grad = Array1::from_elem(weights.len(), 0.0);
    let mut intercept_grad = 0.0;

    let mut shuffled_indexes: Vec<usize> = (0..n).collect();
    shuffled_indexes.shuffle(&mut rand::thread_rng());
    let batch_indexes = shuffled_indexes[0..batch_size].to_vec();

    for i in batch_indexes {
        let x_row = match x.select(Axis(0), &[i]).into_shape_clone((x.dim().1,)) {
            Ok(row) => row,
            Err(_) => panic!("Failed to get a string by index {i} from the array")
        };
        weights_grad += &(-(2.0 / n as f64) * &x_row * (y[i] - (x_row.dot(weights) + *intercept)));
        intercept_grad += -(2.0 / n as f64) * (y[i] - (x_row.dot(weights) + *intercept));
    }

    let new_weights = weights.clone() - weights_grad * learning_rate;
    let new_intercept = *intercept - intercept_grad * learning_rate;

    (new_weights, new_intercept)
}