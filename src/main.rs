use micronn::lin_reg::LinearRegression;
use micronn::metrics;
use ndarray::{arr1, arr2};

fn main() {
    let x = arr2(&[[2.0, 3.0], [3.0, 4.0]]);
    let y = arr1(&[2.5, 3.5]);
    let mut lin_reg = LinearRegression::new();
    lin_reg.fit(&x, &y, 0.01, 100);
    // println!("{:?}", lin_reg.weights.clone().unwrap());
    let x_test = arr2(&[[2.0, 3.0], [3.0, 4.0]]);
    let x_pred = lin_reg.predict(&x_test);
    println!("{:?}", &x_pred);
    println!("{}", metrics::mae(&x_pred, &y));
}