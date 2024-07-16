use burn::prelude::*;

use super::features::{Features, FeaturesConfig};

#[cfg(feature = "pretrained")]
use {super::weights::WeightsSource, super::weights_loader};

#[derive(Config)]
pub struct VggFeaturesOnlyConfig {
    pub features: FeaturesConfig,
}

impl VggFeaturesOnlyConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> VggFeaturesOnly<B> {
        VggFeaturesOnly {
            features: self.features.init(device),
        }
    }
}

#[derive(Debug, Module)]
pub struct VggFeaturesOnly<B: Backend> {
    pub features: Features<B>,
}

impl<B: Backend> VggFeaturesOnly<B> {
    #[cfg(feature = "pretrained")]
    pub(crate) fn vgg_from_weights(
        tch_layers: weights_loader::TchLayers,
        weights: WeightsSource,
        features: FeaturesConfig,
        device: &B::Device,
    ) -> (VggFeaturesOnlyConfig, VggFeaturesOnly<B>) {
        use crate::model::weights_loader::load_weights_record_features_only;

        let record = load_weights_record_features_only(tch_layers, &weights, device).unwrap();

        let config = VggFeaturesOnlyConfig { features };

        let model = config.init::<B>(device).load_record(record);

        (config, model)
    }

    #[cfg(feature = "pretrained")]
    pub fn vgg_16_imagenet_1k_features(
        device: &B::Device,
    ) -> (VggFeaturesOnlyConfig, VggFeaturesOnly<B>) {
        Self::vgg_from_weights(
            weights_loader::TchLayers::Vgg16,
            WeightsSource::vgg_16_imagenet_1k_features(),
            FeaturesConfig::vgg_16(false),
            device,
        )
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        self.features.forward(x)
    }
}

#[cfg(test)]
mod test {
    use burn::{
        backend::{ndarray::NdArrayDevice, NdArray},
        prelude::*,
    };

    #[cfg(feature = "pretrained")]
    #[test]
    fn vgg_16_features_pretrained_sanity_check() {
        use crate::model::vgg_features_only::VggFeaturesOnly;

        type B = NdArray;
        let device = NdArrayDevice::default();
        let device = &device;

        let (_, model) = VggFeaturesOnly::<B>::vgg_16_imagenet_1k_features(device);

        let input = Tensor::zeros([1, 3, 224, 224], device);
        let output = model.forward(input);
        assert_eq!(output.shape().dims, [1, 512, 7, 7]);
    }
}
