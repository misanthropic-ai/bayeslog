use bayeslog::qbbn::{
    common::{
        batch_train::do_batch_training,
        resources::ResourceContext,
        setup::{CommandLineOptions, StorageType},
    },
    scenarios::factory::ScenarioMakerFactory,
};
use clap::Parser;
use env_logger;
use std::time::Instant;
use std::env;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Entity counts to benchmark (comma-separated)
    #[arg(long, default_value = "1,5,10,25,50,100")]
    entity_counts: String,

    /// Compare both models (default if no specific model is selected)
    #[arg(long)]
    compare: bool,

    /// Run only standard model
    #[arg(long)]
    standard_only: bool,

    /// Run only torch model
    #[arg(long)]
    torch_only: bool,
}

fn benchmark_entity_count(entity_count: usize, run_standard: bool, run_torch: bool) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n========== BENCHMARKING WITH {} ENTITIES ==========", entity_count);
    
    let scenario_name = "dating_simple".to_string();
    let config = CommandLineOptions {
        scenario_name: scenario_name.clone(),
        test_scenario: None,
        entities_per_domain: entity_count as i32,
        print_training_loss: false,
        test_example: None,
        marginal_output_file: None,
        storage_type: StorageType::InMemory,
        db_path: None,
    };
    
    let mut standard_time = None;
    let mut torch_time = None;
    
    if run_standard {
        // Benchmark standard model
        unsafe { env::set_var("BAYESLOG_USE_TORCH", "0"); }
        let resources_std = ResourceContext::new(&config)?;
        let scenario_maker = ScenarioMakerFactory::new_shared(&config.scenario_name)?;
        
        println!("\n--- Standard ExponentialModel ---");
        let start_std = Instant::now();
        scenario_maker.setup_scenario(&resources_std)?;
        do_batch_training(&resources_std, scenario_name.clone(), false)?;
        let duration_std = start_std.elapsed();
        println!("Time taken: {:.3} seconds", duration_std.as_secs_f64());
        standard_time = Some(duration_std);
    }
    
    if run_torch {
        // Benchmark torch model
        unsafe { env::set_var("BAYESLOG_USE_TORCH", "1"); }
        let resources_torch = ResourceContext::new(&config)?;
        let scenario_maker = ScenarioMakerFactory::new_shared(&config.scenario_name)?;
        
        println!("\n--- TorchExponentialModel (GPU) ---");
        let start_torch = Instant::now();
        scenario_maker.setup_scenario(&resources_torch)?;
        do_batch_training(&resources_torch, scenario_name.clone(), true)?;
        let duration_torch = start_torch.elapsed();
        println!("Time taken: {:.3} seconds", duration_torch.as_secs_f64());
        torch_time = Some(duration_torch);
    }
    
    // Calculate speedup if both were run
    if let (Some(std_time), Some(torch_time)) = (standard_time, torch_time) {
        let speedup = std_time.as_secs_f64() / torch_time.as_secs_f64();
        println!("\n--- Performance Summary ---");
        println!("Standard model: {:.3}s", std_time.as_secs_f64());
        println!("Torch model: {:.3}s", torch_time.as_secs_f64());
        println!("Speedup: {:.2}x", speedup);
        
        if speedup > 1.0 {
            println!("Torch model is {:.1}% faster", (speedup - 1.0) * 100.0);
        } else {
            println!("Standard model is {:.1}% faster", (1.0 / speedup - 1.0) * 100.0);
        }
    }
    
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let args = Args::parse();
    
    // Parse entity counts
    let entity_counts: Vec<usize> = args.entity_counts
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    
    if entity_counts.is_empty() {
        eprintln!("No valid entity counts provided");
        return Ok(());
    }
    
    // Determine which models to run
    let run_standard = args.standard_only || args.compare || (!args.standard_only && !args.torch_only);
    let run_torch = args.torch_only || args.compare || (!args.standard_only && !args.torch_only);
    
    println!("=== BAYESLOG MODEL PERFORMANCE BENCHMARK ===");
    if run_standard && run_torch {
        println!("Comparing Standard ExponentialModel vs TorchExponentialModel");
    } else if run_standard {
        println!("Benchmarking Standard ExponentialModel only");
    } else if run_torch {
        println!("Benchmarking TorchExponentialModel only");
    }
    println!("Entity counts to test: {:?}", entity_counts);
    
    for &count in &entity_counts {
        if let Err(e) = benchmark_entity_count(count, run_standard, run_torch) {
            eprintln!("Error benchmarking {} entities: {}", count, e);
        }
    }
    
    println!("\n=== BENCHMARK COMPLETE ===");
    Ok(())
}