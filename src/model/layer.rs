use burn::{config::Config, prelude::*};
use nn::{
    conv::{Conv2d, Conv2dConfig},
    pool::{MaxPool2d, MaxPool2dConfig},
    BatchNorm, BatchNormConfig,
};

use crate::relu::ReLU;

#[derive(Debug, Config)]
pub struct LayerConfig {
    pub conv: Conv2dConfig,
    pub bn: Option<BatchNormConfig>,
    pub pool: Option<MaxPool2dConfig>,
}

impl LayerConfig {
    pub fn init<B: Backend>(&self, device: &Device<B>) -> Layer<B> {
        Layer {
            conv: self.conv.init(device),
            bn: self.bn.as_ref().map(|bn| bn.init(device)),
            relu: Default::default(),
            pool: self.pool.as_ref().map(|pool| pool.init()),
        }
    }
}

#[derive(Debug, Module)]
pub struct Layer<B: Backend> {
    pub conv: Conv2d<B>,
    pub bn: Option<BatchNorm<B, 2>>,
    pub relu: ReLU,
    pub pool: Option<MaxPool2d>,
}

impl<B: Backend> Layer<B> {
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.conv.forward(x);
        let x = if let Some(bn) = &self.bn {
            bn.forward(x)
        } else {
            x
        };
        let x = self.relu.forward(x);
        let x = if let Some(pool) = &self.pool {
            pool.forward(x)
        } else {
            x
        };
        x
    }
}
