use anyhow::{Result, bail};
use burn::tensor::Device;
use std::{fmt, str::FromStr};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendKind {
    Wgpu,
    Cuda,
    Flex,
}
impl fmt::Display for BackendKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Wgpu => write!(f, "wgpu"),
            Self::Cuda => write!(f, "cuda"),
            Self::Flex => write!(f, "flex"),
        }
    }
}
impl FromStr for BackendKind {
    type Err = anyhow::Error;
    fn from_str(value: &str) -> Result<Self> {
        match value.to_ascii_lowercase().as_str() {
            "wgpu" => Ok(Self::Wgpu),
            "cuda" => Ok(Self::Cuda),
            "flex" | "cpu" => Ok(Self::Flex),
            other => bail!("Unknown backend '{other}'. Choose wgpu, cuda, or flex."),
        }
    }
}
/// Creates exactly the backend requested by the user. There is no automatic fallback.
pub fn create_device(backend: BackendKind) -> Result<Device> {
    match backend {
        BackendKind::Wgpu => {
            #[cfg(feature = "backend-wgpu")]
            {
                Ok(Device::wgpu(Default::default()))
            }
            #[cfg(not(feature = "backend-wgpu"))]
            {
                bail!("WGPU was not compiled. Rebuild with --features backend-wgpu.")
            }
        }
        BackendKind::Cuda => {
            #[cfg(feature = "backend-cuda")]
            {
                Ok(Device::cuda(0))
            }
            #[cfg(not(feature = "backend-cuda"))]
            {
                bail!("CUDA was not compiled. Rebuild with --features backend-cuda.")
            }
        }
        BackendKind::Flex => {
            #[cfg(feature = "backend-flex")]
            {
                Ok(Device::flex())
            }
            #[cfg(not(feature = "backend-flex"))]
            {
                bail!("Flex was not compiled. Rebuild with --features backend-flex.")
            }
        }
    }
}
pub fn training_device(device: Device) -> Device {
    device.autodiff()
}
