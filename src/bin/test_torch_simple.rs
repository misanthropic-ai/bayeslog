use bayeslog::qbbn::model::device::{select_device, TorchConfig};
use log::info;
use std::time::Instant;
use tch::{nn, nn::OptimizerConfig, Tensor, Kind};

fn test_basic_torch_operations() -> Result<(), Box<dyn std::error::Error>> {
    info!("Testing basic PyTorch operations...");
    
    // Test device selection
    let device = select_device();
    info!("Selected device: {:?}", device);
    
    // Test tensor creation
    let x = Tensor::randn(&[3, 3], (Kind::Float, device));
    info!("Created random tensor with shape: {:?}", x.size());
    
    // Test tensor operations
    let y = Tensor::randn(&[3, 3], (Kind::Float, device));
    let z = &x + &y;
    info!("Added two tensors, result shape: {:?}", z.size());
    
    // Test matrix multiplication
    let result = x.matmul(&y);
    info!("Matrix multiplication result shape: {:?}", result.size());
    
    // Test exponential (used in our model)
    let exp_result = x.exp();
    info!("Exponential result shape: {:?}", exp_result.size());
    
    Ok(())
}

fn test_torch_optimizer() -> Result<(), Box<dyn std::error::Error>> {
    info!("\nTesting PyTorch optimizer...");
    
    let device = select_device();
    let config = TorchConfig::default();
    
    // Create a simple model (just a weight vector)
    let mut vs = nn::VarStore::new(device);
    let weights = vs.root().var("weights", &[10], nn::Init::Randn { mean: 0.0, stdev: 1.0 });
    
    // Create optimizer
    let mut optimizer = nn::Adam::default().build(&vs, config.learning_rate)?;
    
    // Simulate training steps
    info!("Running 10 optimization steps...");
    let start = Instant::now();
    
    for step in 0..10 {
        // Create fake loss (sum of squares)
        let loss = weights.pow_tensor_scalar(2).sum(Kind::Float);
        
        // Backward pass
        optimizer.zero_grad();
        loss.backward();
        optimizer.step();
        
        if step % 2 == 0 {
            let loss_value: f64 = loss.double_value(&[]);
            info!("Step {}: loss = {:.6}", step, loss_value);
        }
    }
    
    info!("Optimization completed in {:?}", start.elapsed());
    
    Ok(())
}

fn benchmark_tensor_operations() -> Result<(), Box<dyn std::error::Error>> {
    info!("\nBenchmarking tensor operations...");
    
    let device = select_device();
    let sizes = vec![10, 100, 1000];
    
    for size in sizes {
        info!("\nTesting with size {}x{}", size, size);
        
        // Create tensors
        let a = Tensor::randn(&[size, size], (Kind::Float, device));
        let b = Tensor::randn(&[size, size], (Kind::Float, device));
        
        // Benchmark matrix multiplication
        let start = Instant::now();
        let _c = a.matmul(&b);
        let matmul_time = start.elapsed();
        
        // Benchmark element-wise operations
        let start = Instant::now();
        let _d = (&a + &b).exp();
        let elementwise_time = start.elapsed();
        
        info!("  Matrix multiplication: {:?}", matmul_time);
        info!("  Element-wise ops: {:?}", elementwise_time);
    }
    
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    
    info!("=== Testing PyTorch Integration ===");
    
    // Basic operations
    test_basic_torch_operations()?;
    
    // Optimizer test
    test_torch_optimizer()?;
    
    // Benchmarks
    benchmark_tensor_operations()?;
    
    info!("\n=== All PyTorch tests completed successfully! ===");
    Ok(())
}