pub mod classifier;
pub mod features;
pub mod layer;
pub mod vgg_features_only;
pub mod weights;

#[cfg(feature = "pretrained")]
pub mod weights_loader;

use burn::prelude::*;
use classifier::{Classifier, ClassifierConfig};
use features::{Features, FeaturesConfig};

use nn::pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig};
#[cfg(feature = "pretrained")]
use weights::WeightsSource;

fn vgg_pool() -> AdaptiveAvgPool2dConfig {
    AdaptiveAvgPool2dConfig::new([7, 7])
}

#[derive(Config)]
pub struct VggConfig {
    pub features: FeaturesConfig,
    pub avgpool: AdaptiveAvgPool2dConfig,
    pub classifier: ClassifierConfig,
}

impl VggConfig {
    pub fn vgg_11(use_bn: bool) -> Self {
        Self {
            features: FeaturesConfig::vgg_11(use_bn),
            avgpool: vgg_pool(),
            classifier: ClassifierConfig::vgg_classifier(0.5, 1000),
        }
    }

    pub fn vgg_13(use_bn: bool) -> Self {
        Self {
            features: FeaturesConfig::vgg_13(use_bn),
            avgpool: vgg_pool(),
            classifier: ClassifierConfig::vgg_classifier(0.5, 1000),
        }
    }

    pub fn vgg_16(use_bn: bool) -> Self {
        Self {
            features: FeaturesConfig::vgg_16(use_bn),
            avgpool: vgg_pool(),
            classifier: ClassifierConfig::vgg_classifier(0.5, 1000),
        }
    }

    pub fn vgg_19(use_bn: bool) -> Self {
        Self {
            features: FeaturesConfig::vgg_19(use_bn),
            avgpool: vgg_pool(),
            classifier: ClassifierConfig::vgg_classifier(0.5, 1000),
        }
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> Vgg<B> {
        Vgg {
            features: self.features.init(device),
            avgpool: self.avgpool.init(),
            classifier: self.classifier.init(device),
        }
    }
}

#[derive(Debug, Module)]
pub struct Vgg<B: Backend> {
    pub features: Features<B>,
    pub avgpool: AdaptiveAvgPool2d,
    pub classifier: Classifier<B>,
}

impl<B: Backend> Vgg<B> {
    #[cfg(feature = "pretrained")]
    pub(crate) fn vgg_from_weights(
        tch_layers: weights_loader::TchLayers,
        weights: WeightsSource,
        features: FeaturesConfig,
        dropout: f64,
        num_classes: usize,
        device: &B::Device,
    ) -> (VggConfig, Vgg<B>) {
        use weights_loader::load_weights_record;

        let record = load_weights_record(tch_layers, &weights, device).unwrap();

        let config = VggConfig {
            features,
            avgpool: vgg_pool(),
            classifier: ClassifierConfig::vgg_classifier(dropout, num_classes),
        };

        let model = config.init::<B>(device).load_record(record);

        (config, model)
    }

    #[cfg(feature = "pretrained")]
    pub fn vgg_11_imagenet_1k(device: &B::Device) -> (VggConfig, Vgg<B>) {
        Self::vgg_from_weights(
            weights_loader::TchLayers::Vgg11,
            WeightsSource::vgg_11_imagenet_1k(),
            FeaturesConfig::vgg_11(false),
            0.5,
            1000,
            device,
        )
    }

    #[cfg(feature = "pretrained")]
    pub fn vgg_11_bn_imagenet_1k(device: &B::Device) -> (VggConfig, Vgg<B>) {
        Self::vgg_from_weights(
            weights_loader::TchLayers::Vgg11,
            WeightsSource::vgg_11_bn_imagenet_1k(),
            FeaturesConfig::vgg_11(true),
            0.5,
            1000,
            device,
        )
    }

    #[cfg(feature = "pretrained")]
    pub fn vgg_13_imagenet_1k(device: &B::Device) -> (VggConfig, Vgg<B>) {
        Self::vgg_from_weights(
            weights_loader::TchLayers::Vgg13,
            WeightsSource::vgg_13_imagenet_1k(),
            FeaturesConfig::vgg_13(false),
            0.5,
            1000,
            device,
        )
    }

    #[cfg(feature = "pretrained")]
    pub fn vgg_13_bn_imagenet_1k(device: &B::Device) -> (VggConfig, Vgg<B>) {
        Self::vgg_from_weights(
            weights_loader::TchLayers::Vgg13,
            WeightsSource::vgg_13_bn_imagenet_1k(),
            FeaturesConfig::vgg_13(true),
            0.5,
            1000,
            device,
        )
    }

    #[cfg(feature = "pretrained")]
    pub fn vgg_16_imagenet_1k(device: &B::Device) -> (VggConfig, Vgg<B>) {
        Self::vgg_from_weights(
            weights_loader::TchLayers::Vgg16,
            WeightsSource::vgg_16_imagenet_1k(),
            FeaturesConfig::vgg_16(false),
            0.5,
            1000,
            device,
        )
    }

    #[cfg(feature = "pretrained")]
    pub fn vgg_16_bn_imagenet_1k(device: &B::Device) -> (VggConfig, Vgg<B>) {
        Self::vgg_from_weights(
            weights_loader::TchLayers::Vgg16,
            WeightsSource::vgg_16_bn_imagenet_1k(),
            FeaturesConfig::vgg_16(true),
            0.5,
            1000,
            device,
        )
    }

    #[cfg(feature = "pretrained")]
    pub fn vgg_19_imagenet_1k(device: &B::Device) -> (VggConfig, Vgg<B>) {
        Self::vgg_from_weights(
            weights_loader::TchLayers::Vgg19,
            WeightsSource::vgg_19_imagenet_1k(),
            FeaturesConfig::vgg_19(false),
            0.5,
            1000,
            device,
        )
    }

    #[cfg(feature = "pretrained")]
    pub fn vgg_19_bn_imagenet_1k(device: &B::Device) -> (VggConfig, Vgg<B>) {
        Self::vgg_from_weights(
            weights_loader::TchLayers::Vgg19,
            WeightsSource::vgg_19_bn_imagenet_1k(),
            FeaturesConfig::vgg_19(true),
            0.5,
            1000,
            device,
        )
    }

    pub fn forward_features(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        self.features.forward(x)
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 2> {
        let x = self.features.forward(x);
        let x = self.avgpool.forward(x);
        let x = x.flatten::<2>(1, 3);
        let x = self.classifier.forward(x);
        x
    }
}

#[cfg(test)]
mod test {
    use burn::{
        backend::{ndarray::NdArrayDevice, NdArray},
        prelude::*,
    };

    use super::VggConfig;

    use crate::model::Vgg;

    fn sanity_check(config: VggConfig) {
        type B = NdArray;
        let device = NdArrayDevice::default();
        let device = &device;

        let input: Tensor<B, 4> = Tensor::ones([4, 3, 224, 224], device);

        let model = config.init::<B>(device);

        let output = model.forward_features(input.clone());
        assert_eq!(output.shape().dims, [4, 512, 7, 7]);

        let output = model.forward(input.clone());
        assert_eq!(output.shape().dims, [4, 1000]);
    }

    #[test]
    fn sanity_check_vgg_11() {
        sanity_check(VggConfig::vgg_11(false))
    }

    #[test]
    fn sanity_check_vgg_13() {
        sanity_check(VggConfig::vgg_13(false))
    }

    #[test]
    fn sanity_check_vgg_16() {
        sanity_check(VggConfig::vgg_16(false))
    }

    #[test]
    fn sanity_check_vgg_19() {
        sanity_check(VggConfig::vgg_19(false))
    }

    /// Test input data for pre-trained models
    ///
    /// ```python
    /// input = torch.stack(
    ///     [
    ///         torch.ones((3, 299, 299)) * 0.0,
    ///         torch.ones((3, 299, 299)) * 0.5,
    ///         torch.ones((3, 299, 299)) * 1.0,
    ///         torch.linspace(0.0, 1.0, 3 * 299 * 299).reshape((3, 299, 299)),
    ///     ]
    /// )
    /// ```
    fn test_data<B: Backend>(device: &B::Device) -> Tensor<B, 4> {
        let linspace = {
            let mut v = vec![0.0; 3 * 224 * 224];
            for i in 0..3 * 224 * 224 {
                v[i] = i as f32 / (3.0 * 224.0 * 224.0 - 1.0);
            }
            v
        };

        let input = Tensor::stack::<4>(
            [
                Tensor::<B, 3>::ones([3, 224, 224], device) * 0.0,
                Tensor::<B, 3>::ones([3, 224, 224], device) * 0.5,
                Tensor::<B, 3>::ones([3, 224, 224], device) * 1.0,
                Tensor::<B, 1>::from_floats(linspace.as_slice(), device)
                    .reshape::<3, _>([3, 224, 224]),
            ]
            .to_vec(),
            0,
        );

        input
    }

    fn test_image_data<B: Backend>(device: &B::Device) -> Tensor<B, 4> {
        let img = image::io::Reader::open("./res/imagenet-car-224x224.png")
            .unwrap()
            .decode()
            .unwrap();
        let img = img.to_rgb8();

        // let data = TensorData::new(vec![], [1, 3, 224, 224]);
        let mut data = vec![0.0f32; 3 * 224 * 224];
        for (x, y, pixel) in img.enumerate_pixels() {
            let row = y as usize;
            let col = x as usize;
            data[0 * (224 * 224) + row * 224 + col] = (pixel.0[0] as f32) / 255.0 - 0.5;
            data[1 * (224 * 224) + row * 224 + col] = (pixel.0[1] as f32) / 255.0 - 0.5;
            data[2 * (224 * 224) + row * 224 + col] = (pixel.0[2] as f32) / 255.0 - 0.5;
        }
        let data = TensorData::new(data, [1, 3, 224, 224]);
        Tensor::from_data(data, device)
    }

    /// Comparing the output of the Burn and PyTorch versions of VGG.
    ///
    /// The expected output was generated using the following Python script:
    ///
    ///```python
    /// import torch
    /// from torchvision.models import VGG11_Weights, VGG13_Weights, VGG16_Weights, VGG19_Weights, vgg11, vgg13, vgg16, vgg19
    ///
    /// model = vgg11(VGG11_Weights)
    /// model.eval()
    ///
    /// output = model(test_data()) # see above
    /// print(output[:, :3])
    /// ```py
    #[cfg(feature = "pretrained")]
    fn test_pretrained_img<B: Backend>(model: &Vgg<B>, target: [f32; 3], device: &B::Device) {
        let input = test_image_data(device);
        let output = model.forward(input);

        let target = Tensor::<B, 2>::from_floats([target], device);

        let output = output.slice([0..1, 0..3]).to_data();

        let target = target.to_data();
        output.assert_approx_eq(&target, 1);
    }

    /// Comparing the output of the Burn and PyTorch versions of VGG.
    ///
    /// The expected output was generated using the following Python script:
    ///
    ///```python
    /// import torch
    /// from torchvision.models import VGG11_Weights, VGG13_Weights, VGG16_Weights, VGG19_Weights, vgg11, vgg13, vgg16, vgg19
    ///
    /// model = vgg11(VGG11_Weights)
    /// model.eval()
    ///
    /// output = model(test_data()) # see above
    /// print(output[:, :3])
    /// ```py
    #[cfg(feature = "pretrained")]
    fn test_pretrained<B: Backend>(model: &Vgg<B>, target: [[f32; 3]; 4], device: &B::Device) {
        let output = model.forward(test_data(device));

        let target = Tensor::<B, 2>::from_floats(target, device);

        let output = output.slice([0..4, 0..3]).to_data();

        let target = target.to_data();
        output.assert_approx_eq(&target, 1);
    }

    /// Comparing the output of the Burn and PyTorch versions of Vgg11.
    ///
    /// In Python, this image data can be loaded as follows:
    /// ```python
    /// from PIL import Image
    ///
    /// image_path = "imagenet-car-224x224.png"
    /// image = Image.open(image_path)
    /// preprocess = transforms.Compose(
    ///     [
    ///         # transforms.Resize((224, 224)),
    ///         transforms.ToTensor(),
    ///     ]
    /// )
    /// image_tensor = preprocess(image)
    /// assert type(image_tensor) == torch.Tensor
    /// image_tensor = image_tensor[:3, :, :].unsqueeze(0)  # Shape: [1, 3, 224, 224]
    /// # image_tensor is already normalized to [0, 1]
    /// # now, normalize it to [-0.5, 0.5]
    /// image_tensor = image_tensor - 0.5
    /// ```
    #[test]
    #[cfg(feature = "pretrained")]
    fn vgg_11_pretrained() {
        type B = NdArray;
        let device = NdArrayDevice::default();
        let device = &device;
        let (_, model) = Vgg::<B>::vgg_11_imagenet_1k(device);
        test_pretrained::<B>(
            &model,
            [
                [-0.2929, 1.0541, -0.8971],
                [-0.6121, 1.0163, -1.4320],
                [-0.5993, 0.6191, -1.4677],
                [-0.5747, 0.9056, -0.4442],
            ],
            device,
        );
        test_pretrained_img::<B>(&model, [-0.7460, -0.4649, -1.6900], device);
    }

    /// Comparing the output of the Burn and PyTorch versions of Vgg13.
    #[test]
    #[cfg(feature = "pretrained")]
    fn vgg_13_pretrained() {
        type B = NdArray;
        let device = NdArrayDevice::default();
        let device = &device;
        let (_, model) = Vgg::<B>::vgg_13_imagenet_1k(device);
        test_pretrained::<B>(
            &model,
            [
                [-0.0710, 0.9949, -1.4211],
                [-0.2922, 0.6214, -1.6742],
                [-0.3164, 0.5477, -1.7871],
                [-0.3703, 0.3363, -1.1878],
            ],
            device,
        );
        test_pretrained_img::<B>(&model, [-1.9690, -0.9368, -1.4227], device);
    }

    /// Comparing the output of the Burn and PyTorch versions of Vgg16.
    #[test]
    #[cfg(feature = "pretrained")]
    fn vgg_16_pretrained() {
        type B = NdArray;
        let device = NdArrayDevice::default();
        let device = &device;
        let (_, model) = Vgg::<B>::vgg_16_imagenet_1k(device);
        test_pretrained::<B>(
            &model,
            [
                [-0.5907, 0.8917, -1.6282],
                [-0.9842, 0.7330, -2.0050],
                [-1.1006, 0.4054, -2.2822],
                [-0.6032, 0.4775, -0.9045],
            ],
            device,
        );
        test_pretrained_img::<B>(&model, [-1.3699, -1.0582, -0.6606], device);
    }

    /// Comparing the output of the Burn and PyTorch versions of Vgg19.
    #[test]
    #[cfg(feature = "pretrained")]
    fn vgg_19_pretrained() {
        type B = NdArray;
        let device = NdArrayDevice::default();
        let device = &device;
        let (_, model) = Vgg::<B>::vgg_19_imagenet_1k(device);
        test_pretrained::<B>(
            &model,
            [
                [-0.5111, 1.1041, -1.5981],
                [-0.7502, 0.9746, -1.8149],
                [-0.8524, 0.8239, -2.0404],
                [-0.5093, 0.4800, -0.5323],
            ],
            device,
        );
        test_pretrained_img::<B>(&model, [-1.5073, -2.7802, -2.3567], device);
    }
}
