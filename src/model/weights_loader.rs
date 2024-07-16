use {
    super::{vgg_features_only::VggFeaturesOnlyRecord, VggRecord},
    burn::{
        prelude::*,
        record::{FullPrecisionSettings, Recorder, RecorderError},
        tensor::Device,
    },
    burn_import::pytorch::{LoadArgs, PyTorchFileRecorder},
};

pub(crate) enum TchLayers {
    Vgg11,
    Vgg13,
    Vgg16,
    Vgg19,
}

impl TchLayers {
    fn tch_feature_layers(self) -> &'static [usize] {
        match self {
            Self::Vgg11 => &[0, 3, 6, 8, 11, 13, 16, 18],
            Self::Vgg13 => &[0, 2, 5, 7, 10, 12, 15, 17, 20, 22],
            Self::Vgg16 => &[0, 2, 5, 7, 10, 12, 14, 17, 19, 21, 24, 26, 28],
            Self::Vgg19 => &[0, 2, 5, 7, 10, 12, 14, 16, 19, 21, 23, 25, 28, 30, 32, 34],
        }
    }
}

/// Load specified pre-trained PyTorch weights as a record.
pub(crate) fn load_weights_record<B: Backend>(
    tch_layers: TchLayers,
    weights: &super::weights::WeightsSource,
    device: &Device<B>,
) -> Result<VggRecord<B>, RecorderError> {
    let mut load_args = LoadArgs::new(weights.file_path.clone());

    for (i, tch_layer) in tch_layers.tch_feature_layers().into_iter().enumerate() {
        load_args = load_args.with_key_remap(
            &format!("features\\.{tch_layer}\\.(.+)"),
            &format!("features.layers.{i}.conv.$1"),
        );
    }

    for (tch_layer, burn_layer) in [(0, 1), (3, 2), (6, 3)] {
        load_args = load_args.with_key_remap(
            &format!("classifier.{tch_layer}.(.+)"),
            &format!("classifier.linear_{burn_layer}.$1"),
        );
    }

    let record = PyTorchFileRecorder::<FullPrecisionSettings>::new()
        .load(load_args.with_debug_print(), device)?;

    Ok(record)
}

/// Load specified pre-trained PyTorch weights as a record.
pub(crate) fn load_weights_record_features_only<B: Backend>(
    tch_layers: TchLayers,
    weights: &super::weights::WeightsSource,
    device: &Device<B>,
) -> Result<VggFeaturesOnlyRecord<B>, RecorderError> {
    let mut load_args = LoadArgs::new(weights.file_path.clone());

    for (i, tch_layer) in tch_layers.tch_feature_layers().into_iter().enumerate() {
        load_args = load_args.with_key_remap(
            &format!("features\\.{tch_layer}\\.(.+)"),
            &format!("features.layers.{i}.conv.$1"),
        );
    }

    let record = PyTorchFileRecorder::<FullPrecisionSettings>::new()
        .load(load_args.with_debug_print(), device)?;

    Ok(record)
}
