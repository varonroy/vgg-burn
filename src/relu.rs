use burn::prelude::*;

#[derive(Debug, Clone, Module, Default)]
pub struct ReLU {}

impl ReLU {
    pub fn forward<B: Backend, const N: usize>(&self, x: Tensor<B, N>) -> Tensor<B, N> {
        burn::tensor::activation::relu(x)
    }
}
