use bayeslog::qbbn::model::device::select_device;
use log::info;

fn main() {
    env_logger::init();
    
    info!("Testing PyTorch device selection...");
    let device = select_device();
    info!("Selected device: {:?}", device);
    
    // Test tensor creation
    info!("Testing tensor creation...");
    let tensor = tch::Tensor::randn(&[3, 3], (tch::Kind::Float, device));
    info!("Created tensor with shape: {:?}", tensor.size());
    
    // Test MPS availability
    #[cfg(target_os = "macos")]
    {
        info!("MPS available: {}", tch::utils::has_mps());
    }
    
    info!("PyTorch test completed successfully!");
}