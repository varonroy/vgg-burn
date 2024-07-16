use burn::{config::Config, prelude::*};
use nn::{conv::Conv2dConfig, pool::MaxPool2dConfig, BatchNormConfig};

use super::layer::{Layer, LayerConfig};

#[derive(Debug, Clone, Copy)]
pub(crate) enum Template {
    V(usize),
    VM(usize),
}

const fn v(channels: usize) -> Template {
    Template::V(channels)
}

const fn vm(channels: usize) -> Template {
    Template::VM(channels)
}

#[rustfmt::skip]
pub (crate)const VGG_11_TEMPLATE: [Template; 8] = [vm(64), vm(128), v(256), vm(256), v(512), vm(512), v(512), vm(512)];
#[rustfmt::skip]
pub(crate)const VGG_13_TEMPLATE: [Template; 10] = [v(64), vm(64), v(128), vm(128), v(256), vm(256), v(512), vm(512), v(512), vm(512)];
#[rustfmt::skip]
pub (crate)const VGG_16_TEMPLATE: [Template; 13] = [v(64), vm(64), v(128),vm(128),v(256), v(256),vm(256),v(512), v(512),vm(512),v(512), v(512),vm(512)];
#[rustfmt::skip]
pub (crate)const VGG_19_TEMPLATE: [Template; 16] = [v(64), vm(64), v(128),vm(128),v(256), v(256),v(256), vm(256), v(512),v(512), v(512),vm(512),v(512), v(512),v(512), vm(512)];

fn template_to_vgg(tempalte: &[Template], use_bn: bool) -> Vec<LayerConfig> {
    let mut in_channels = 3;
    let mut layers = Vec::new();
    for t in tempalte {
        let (channels, pool) = match *t {
            Template::VM(channels) => (channels, true),
            Template::V(channels) => (channels, false),
        };
        let conv = Conv2dConfig::new([in_channels, channels], [3, 3])
            .with_padding(nn::PaddingConfig2d::Explicit(1, 1));
        let bn = use_bn.then(|| BatchNormConfig::new(channels));

        layers.push(LayerConfig {
            conv,
            bn,
            pool: pool.then_some(MaxPool2dConfig::new([2, 2]).with_strides([2, 2])),
        });

        in_channels = channels;
    }
    layers
}

#[derive(Debug, Config)]
pub struct FeaturesConfig {
    pub layers: Vec<LayerConfig>,
}

impl FeaturesConfig {
    pub fn vgg_11(use_bn: bool) -> Self {
        Self {
            layers: template_to_vgg(&VGG_11_TEMPLATE, use_bn),
        }
    }

    pub fn vgg_13(use_bn: bool) -> Self {
        Self {
            layers: template_to_vgg(&VGG_13_TEMPLATE, use_bn),
        }
    }

    pub fn vgg_16(use_bn: bool) -> Self {
        Self {
            layers: template_to_vgg(&VGG_16_TEMPLATE, use_bn),
        }
    }

    pub fn vgg_19(use_bn: bool) -> Self {
        Self {
            layers: template_to_vgg(&VGG_19_TEMPLATE, use_bn),
        }
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> Features<B> {
        Features {
            layers: self.layers.iter().map(|layer| layer.init(device)).collect(),
        }
    }
}

#[derive(Debug, Module)]
pub struct Features<B: Backend> {
    pub layers: Vec<Layer<B>>,
}

impl<B: Backend> Features<B> {
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        self.layers.iter().fold(x, |x, layer| layer.forward(x))
    }
}
