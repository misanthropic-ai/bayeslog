use log::{info, warn};
use std::env;
use tch::Device;

/// Selects the best available device for PyTorch operations
/// Priority: Environment variable > MPS (macOS) > CUDA > CPU
pub fn select_device() -> Device {
    // Check environment variable first
    if let Ok(device_str) = env::var("BAYESLOG_DEVICE") {
        match device_str.to_lowercase().as_str() {
            "cpu" => {
                info!("Using CPU device (from environment variable)");
                return Device::Cpu;
            }
            "cuda" | "gpu" => {
                if tch::Cuda::is_available() {
                    info!("Using CUDA device (from environment variable)");
                    return Device::Cuda(0);
                } else {
                    warn!("CUDA requested but not available, falling back to CPU");
                    return Device::Cpu;
                }
            }
            "mps" => {
                if tch::utils::has_mps() {
                    info!("Using MPS device (from environment variable)");
                    return Device::Mps;
                } else {
                    warn!("MPS requested but not available, falling back to CPU");
                    return Device::Cpu;
                }
            }
            _ => {
                warn!("Unknown device '{}' in BAYESLOG_DEVICE, using auto-detection", device_str);
            }
        }
    }

    // Auto-detect best available device
    #[cfg(target_os = "macos")]
    {
        if tch::utils::has_mps() {
            info!("Using MPS device (Metal Performance Shaders on macOS)");
            return Device::Mps;
        }
    }

    if tch::Cuda::is_available() {
        info!("Using CUDA device");
        return Device::Cuda(0);
    }

    info!("Using CPU device");
    Device::Cpu
}

/// Configuration for the PyTorch-based model
#[derive(Debug, Clone)]
pub struct TorchConfig {
    pub device: Device,
    pub learning_rate: f64,
    pub batch_size: usize,
    pub optimizer: OptimizerType,
    pub max_epochs: usize,
    pub early_stopping_patience: usize,
    pub gradient_clip: Option<f64>,
}

#[derive(Debug, Clone, Copy)]
pub enum OptimizerType {
    SGD { momentum: f64 },
    Adam { beta1: f64, beta2: f64, epsilon: f64 },
    AdamW { beta1: f64, beta2: f64, epsilon: f64, weight_decay: f64 },
}

impl Default for TorchConfig {
    fn default() -> Self {
        Self {
            device: select_device(),
            learning_rate: 0.001,
            batch_size: 32,
            optimizer: OptimizerType::Adam {
                beta1: 0.9,
                beta2: 0.999,
                epsilon: 1e-8,
            },
            max_epochs: 100,
            early_stopping_patience: 10,
            gradient_clip: Some(1.0),
        }
    }
}

impl TorchConfig {
    /// Create config from environment variables
    pub fn from_env() -> Self {
        let mut config = Self::default();
        
        if let Ok(lr) = env::var("BAYESLOG_LEARNING_RATE") {
            if let Ok(lr_val) = lr.parse::<f64>() {
                config.learning_rate = lr_val;
            }
        }
        
        if let Ok(bs) = env::var("BAYESLOG_BATCH_SIZE") {
            if let Ok(bs_val) = bs.parse::<usize>() {
                config.batch_size = bs_val;
            }
        }
        
        if let Ok(opt) = env::var("BAYESLOG_OPTIMIZER") {
            config.optimizer = match opt.to_lowercase().as_str() {
                "sgd" => OptimizerType::SGD { momentum: 0.9 },
                "adam" => OptimizerType::Adam {
                    beta1: 0.9,
                    beta2: 0.999,
                    epsilon: 1e-8,
                },
                "adamw" => OptimizerType::AdamW {
                    beta1: 0.9,
                    beta2: 0.999,
                    epsilon: 1e-8,
                    weight_decay: 0.01,
                },
                _ => config.optimizer,
            };
        }
        
        config
    }
}