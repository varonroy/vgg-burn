use std::path::PathBuf;

/// Pre-trained weights metadata.
#[derive(Debug, Clone)]
pub struct WeightsSource {
    pub file_path: PathBuf,
}

#[cfg(feature = "pretrained")]
impl WeightsSource {
    pub fn download(url: &str) -> Self {
        let file_path = downloader::download(url).unwrap();
        Self { file_path }
    }

    pub fn get(file_name: &str) -> Self {
        let file_path = downloader::default_file_path(file_name);
        Self { file_path }
    }

    pub fn vgg_11_imagenet_1k() -> Self {
        Self::download("https://download.pytorch.org/models/vgg11-8a719046.pth")
    }

    pub fn vgg_11_bn_imagenet_1k() -> Self {
        Self::download("https://download.pytorch.org/models/vgg11_bn-6002323d.pth")
    }

    pub fn vgg_13_imagenet_1k() -> Self {
        Self::download("https://download.pytorch.org/models/vgg13-19584684.pth")
    }

    pub fn vgg_13_bn_imagenet_1k() -> Self {
        Self::download("https://download.pytorch.org/models/vgg13_bn-abd245e5.pth")
    }

    pub fn vgg_16_imagenet_1k() -> Self {
        Self::get("vgg16-397923af.pth")
    }

    pub fn vgg_16_imagenet_1k_features() -> Self {
        Self::get("vgg16_features-amdegroot-88682ab5.pth")
    }

    pub fn vgg_16_bn_imagenet_1k() -> Self {
        Self::get("vgg16_bn-6c64b313.pth")
    }

    pub fn vgg_19_imagenet_1k() -> Self {
        Self::get("vgg19-dcbb9e9d.pth")
    }

    pub fn vgg_19_bn_imagenet_1k() -> Self {
        Self::get("vgg19_bn-c79401a0.pth")
    }
}

#[cfg(feature = "pretrained")]
pub mod downloader {
    use burn::data::network::downloader;
    use std::fs::{create_dir_all, File};
    use std::io::Write;
    use std::path::{Path, PathBuf};

    /// Download the pre-trained weights to a specific directory.
    pub fn download_to(url: &str, model_dir: impl AsRef<Path>) -> Result<PathBuf, std::io::Error> {
        let model_dir = model_dir.as_ref();

        if !model_dir.exists() {
            create_dir_all(&model_dir)?;
        }

        let file_base_name = url.rsplit_once('/').unwrap().1;
        let file_name = model_dir.join(file_base_name);
        if !file_name.exists() {
            // Download file content
            let bytes = downloader::download_file_as_bytes(url, file_base_name);

            // Write content to file
            let mut output_file = File::create(&file_name)?;
            let bytes_written = output_file.write(&bytes)?;

            if bytes_written != bytes.len() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "Failed to write the whole model weights file.",
                ));
            }
        }

        Ok(file_name)
    }

    /// Download the pre-trained weights to the local cache directory.
    pub fn download(url: &str) -> Result<PathBuf, std::io::Error> {
        let model_dir = dirs::home_dir()
            .expect("Should be able to get home directory")
            .join(".cache")
            .join("vgg-burn");
        download_to(url, model_dir)
    }

    pub fn default_file_path(file_name: &str) -> PathBuf {
        dirs::home_dir()
            .expect("Should be able to get home directory")
            .join(".cache")
            .join("vgg-burn")
            .join(file_name)
    }
}
