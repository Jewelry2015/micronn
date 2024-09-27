use ndarray::Array1;

pub fn mse(predictions: &Array1<f64>, labels: &Array1<f64>) -> f64 {
    assert_eq!(labels.len(), predictions.len(), "labels' and predictions' lengths must be the same");

    (predictions - labels).powi(2).sum() / (labels.len() as f64)
}

pub fn rmse(predictions: &Array1<f64>, labels: &Array1<f64>) -> f64 {
    mse(labels, predictions).sqrt()
}

pub fn mae(predictions: &Array1<f64>, labels: &Array1<f64>) -> f64 {
    assert_eq!(labels.dim(), predictions.dim(), "labels' and predictions' lengths must be the same");
    
    (predictions - labels).abs().sum() / (labels.len() as f64)
}