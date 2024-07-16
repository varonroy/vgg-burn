use burn::{
    config::Config,
    nn::{DropoutConfig, LinearConfig},
    prelude::*,
};
use nn::{Dropout, Linear};

use crate::relu::ReLU;

#[derive(Debug, Config)]
pub struct ClassifierConfig {
    pub linear_1: LinearConfig,
    pub dropout_1: DropoutConfig,
    pub linear_2: LinearConfig,
    pub dropout_2: DropoutConfig,
    pub linear_3: LinearConfig,
}

impl ClassifierConfig {
    pub fn vgg_classifier(
        dropout: impl Into<Option<f64>>,
        num_classes: impl Into<Option<usize>>,
    ) -> Self {
        let dropout = dropout.into().unwrap_or(0.5);
        let num_classes = num_classes.into().unwrap_or(1000);

        Self {
            linear_1: LinearConfig::new(512 * 7 * 7, 4096),
            dropout_1: DropoutConfig::new(dropout),
            linear_2: LinearConfig::new(4096, 4096),
            dropout_2: DropoutConfig::new(dropout),
            linear_3: LinearConfig::new(4096, num_classes),
        }
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> Classifier<B> {
        Classifier {
            linear_1: self.linear_1.init(device),
            relu_1: Default::default(),
            dropout_1: self.dropout_1.init(),
            linear_2: self.linear_2.init(device),
            relu_2: Default::default(),
            dropout_2: self.dropout_2.init(),
            linear_3: self.linear_3.init(device),
        }
    }
}

#[derive(Debug, Module)]
pub struct Classifier<B: Backend> {
    pub linear_1: Linear<B>,
    pub relu_1: ReLU,
    pub dropout_1: Dropout,
    pub linear_2: Linear<B>,
    pub relu_2: ReLU,
    pub dropout_2: Dropout,
    pub linear_3: Linear<B>,
}

impl<B: Backend> Classifier<B> {
    pub fn forward<const N: usize>(&self, x: Tensor<B, N>) -> Tensor<B, N> {
        let x = self.linear_1.forward(x);
        let x = self.relu_1.forward(x);
        let x = self.dropout_1.forward(x);

        let x = self.linear_2.forward(x);
        let x = self.relu_2.forward(x);
        let x = self.dropout_2.forward(x);

        let x = self.linear_3.forward(x);
        x
    }
}
