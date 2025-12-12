//! GPU Stress Test
//!
//! Pushes the GPU to its limits by testing:
//! - Maximum dimensions (256)
//! - Large particle counts (up to 100K)
//! - Complex loss functions
//! - Sustained throughput
//!
//! Reports memory usage, throughput, and scaling characteristics.

use std::time::{Duration, Instant};
use temper::ThermodynamicSystem;
use temper::expr::*;
use temper::thermodynamic::LossFunction;

fn main() {
    println!(
        "{}",
        "╔══════════════════════════════════════════════════════════════════════════╗"
    );
    println!(
        "{}",
        "║                     GPU STRESS TEST                                      ║"
    );
    println!(
        "{}",
        "╠══════════════════════════════════════════════════════════════════════════╣"
    );
    println!(
        "{}",
        "║  Testing thermodynamic particle system at extreme scales                 ║"
    );
    println!(
        "{}",
        "║  MAX_DIMENSIONS = 256, particle counts up to 100K                        ║"
    );
    println!(
        "{}",
        "╚══════════════════════════════════════════════════════════════════════════╝\n"
    );

    // Test 1: Particle count scaling
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("TEST 1: PARTICLE COUNT SCALING");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    test_particle_scaling();
    println!();

    // Test 2: Dimension scaling
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("TEST 2: DIMENSION SCALING");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    test_dimension_scaling();
    println!();

    // Test 3: Maximum capacity
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("TEST 3: MAXIMUM CAPACITY (256 dims × 50K particles)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    test_maximum_capacity();
    println!();

    // Test 4: Sustained throughput
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("TEST 4: SUSTAINED THROUGHPUT (10 seconds)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    test_sustained_throughput();
    println!();

    // Test 5: Complex loss function stress
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("TEST 5: COMPLEX LOSS FUNCTION (256-dim neural network)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    test_complex_loss();
    println!();

    // Summary
    println!(
        "{}",
        "╔══════════════════════════════════════════════════════════════════════════╗"
    );
    println!(
        "{}",
        "║                         STRESS TEST COMPLETE                             ║"
    );
    println!(
        "{}",
        "╚══════════════════════════════════════════════════════════════════════════╝"
    );
}

fn test_particle_scaling() {
    let particle_counts = [1_000, 5_000, 10_000, 25_000, 50_000, 100_000];
    let dim = 64; // Fixed dimension
    let steps = 100;

    println!("  Dimensions: {}", dim);
    println!("  Steps per test: {}", steps);
    println!();
    println!(
        "  {:>10} {:>12} {:>12} {:>15} {:>12}",
        "Particles", "Init (ms)", "Time (ms)", "Steps/sec", "Memory (MB)"
    );
    println!("  {}", "-".repeat(65));

    for &count in &particle_counts {
        // Measure initialization
        let init_start = Instant::now();
        let mut system =
            ThermodynamicSystem::with_loss_function(count, dim, 1.0, LossFunction::Rastrigin);
        let init_time = init_start.elapsed();

        // Warmup
        for _ in 0..10 {
            system.step();
        }

        // Measure throughput
        let start = Instant::now();
        for _ in 0..steps {
            system.step();
        }
        let elapsed = start.elapsed();

        let steps_per_sec = steps as f64 / elapsed.as_secs_f64();
        // Memory estimate: particle_count * struct_size (520 bytes for 256 dims)
        // But we're using 64 dims here, so: 64*2 + 4 + 4 = 136 bytes
        // GPU buffer is always sized for MAX_DIMENSIONS, so 520 bytes
        let memory_mb = (count * 520) as f64 / (1024.0 * 1024.0);

        println!(
            "  {:>10} {:>12.1} {:>12.1} {:>15.0} {:>12.1}",
            count,
            init_time.as_secs_f64() * 1000.0,
            elapsed.as_secs_f64() * 1000.0,
            steps_per_sec,
            memory_mb
        );
    }
}

fn test_dimension_scaling() {
    let dimensions = [8, 16, 32, 64, 128, 256];
    let particle_count = 10_000;
    let steps = 100;

    println!("  Particle count: {}", particle_count);
    println!("  Steps per test: {}", steps);
    println!();
    println!(
        "  {:>10} {:>12} {:>12} {:>15}",
        "Dimensions", "Init (ms)", "Time (ms)", "Steps/sec"
    );
    println!("  {}", "-".repeat(55));

    for &dim in &dimensions {
        // Measure initialization
        let init_start = Instant::now();
        let mut system = ThermodynamicSystem::with_loss_function(
            particle_count,
            dim,
            1.0,
            LossFunction::Rastrigin,
        );
        let init_time = init_start.elapsed();

        // Warmup
        for _ in 0..10 {
            system.step();
        }

        // Measure throughput
        let start = Instant::now();
        for _ in 0..steps {
            system.step();
        }
        let elapsed = start.elapsed();

        let steps_per_sec = steps as f64 / elapsed.as_secs_f64();

        println!(
            "  {:>10} {:>12.1} {:>12.1} {:>15.0}",
            dim,
            init_time.as_secs_f64() * 1000.0,
            elapsed.as_secs_f64() * 1000.0,
            steps_per_sec
        );
    }
}

fn test_maximum_capacity() {
    let particle_count = 50_000;
    let dim = 256;
    let steps = 50;

    println!("  Configuration:");
    println!("    Particles: {}", particle_count);
    println!("    Dimensions: {}", dim);
    println!(
        "    Buffer size: {:.1} MB",
        (particle_count * 520) as f64 / (1024.0 * 1024.0)
    );
    println!();

    // Initialize
    println!("  Initializing...");
    let init_start = Instant::now();
    let mut system =
        ThermodynamicSystem::with_loss_function(particle_count, dim, 1.0, LossFunction::Rastrigin);
    let init_time = init_start.elapsed();
    println!("    Init time: {:.1}ms", init_time.as_secs_f64() * 1000.0);

    // Warmup
    println!("  Warming up...");
    for _ in 0..5 {
        system.step();
    }

    // Run test
    println!("  Running {} steps...", steps);
    let start = Instant::now();
    for step in 0..steps {
        system.step();
        if (step + 1) % 10 == 0 {
            let elapsed = start.elapsed();
            let rate = (step + 1) as f64 / elapsed.as_secs_f64();
            print!("\r    Step {}/{} ({:.0} steps/sec)", step + 1, steps, rate);
            use std::io::Write;
            std::io::stdout().flush().ok();
        }
    }
    println!();

    let elapsed = start.elapsed();
    let steps_per_sec = steps as f64 / elapsed.as_secs_f64();

    // Read results
    let particles = system.read_particles();
    let valid_energies: Vec<f32> = particles
        .iter()
        .filter(|p| !p.energy.is_nan() && p.energy.is_finite())
        .map(|p| p.energy)
        .collect();

    let min_energy = valid_energies.iter().cloned().fold(f32::MAX, f32::min);
    let max_energy = valid_energies.iter().cloned().fold(f32::MIN, f32::max);
    let mean_energy = valid_energies.iter().sum::<f32>() / valid_energies.len() as f32;

    println!();
    println!("  Results:");
    println!("    Total time: {:?}", elapsed);
    println!("    Throughput: {:.0} steps/sec", steps_per_sec);
    println!(
        "    Valid particles: {}/{}",
        valid_energies.len(),
        particle_count
    );
    println!("    Energy range: [{:.2}, {:.2}]", min_energy, max_energy);
    println!("    Mean energy: {:.2}", mean_energy);

    // Calculate particles × dimensions × steps per second
    let ops_per_sec = particle_count as f64 * dim as f64 * steps_per_sec;
    println!("    Particle-dim ops/sec: {:.2e}", ops_per_sec);
}

fn test_sustained_throughput() {
    let particle_count = 25_000;
    let dim = 128;
    let duration = Duration::from_secs(10);

    println!("  Configuration:");
    println!("    Particles: {}", particle_count);
    println!("    Dimensions: {}", dim);
    println!("    Duration: {:?}", duration);
    println!();

    let mut system =
        ThermodynamicSystem::with_loss_function(particle_count, dim, 1.0, LossFunction::Rastrigin);

    // Warmup
    for _ in 0..20 {
        system.step();
    }

    println!("  Running sustained test...");
    let start = Instant::now();
    let mut total_steps = 0u64;
    let mut last_report = Instant::now();
    let mut steps_since_report = 0u64;

    while start.elapsed() < duration {
        system.step();
        total_steps += 1;
        steps_since_report += 1;

        // Report every second
        if last_report.elapsed() >= Duration::from_secs(1) {
            let rate = steps_since_report as f64 / last_report.elapsed().as_secs_f64();
            let elapsed = start.elapsed();
            print!(
                "\r    {:>2}s: {:>6} steps/sec, {:>8} total steps",
                elapsed.as_secs(),
                rate as u64,
                total_steps
            );
            use std::io::Write;
            std::io::stdout().flush().ok();
            last_report = Instant::now();
            steps_since_report = 0;
        }
    }
    println!();

    let elapsed = start.elapsed();
    let avg_rate = total_steps as f64 / elapsed.as_secs_f64();

    // Read final state
    let particles = system.read_particles();
    let valid = particles.iter().filter(|p| !p.energy.is_nan()).count();
    let min_energy = particles
        .iter()
        .filter(|p| !p.energy.is_nan())
        .map(|p| p.energy)
        .fold(f32::MAX, f32::min);

    println!();
    println!("  Results:");
    println!("    Total steps: {}", total_steps);
    println!("    Average rate: {:.0} steps/sec", avg_rate);
    println!("    Valid particles: {}/{}", valid, particle_count);
    println!("    Final min energy: {:.4}", min_energy);
}

fn test_complex_loss() {
    let particle_count = 5_000;
    let dim = 200; // Large neural network: 8x16 + 16 + 16x8 + 8 = 280 params (use 200)
    let steps = 200;

    println!("  Testing high-dimensional Griewank function (complex loss)");
    println!("  Configuration:");
    println!("    Particles: {}", particle_count);
    println!("    Dimensions: {} (high-dimensional optimization)", dim);
    println!();

    // Use Griewank function - complex multimodal function with many local minima
    // f(x) = 1 + sum(x_i^2 / 4000) - prod(cos(x_i / sqrt(i+1)))
    let griewank_expr = const_(1.0) + sum_dims(|x, _| x.powi(2) / 4000.0)
        - prod_dims(|x, i| cos(x / sqrt(i + 1.0)));

    println!(
        "  Initializing with Griewank function ({} dimensions)...",
        dim
    );
    let init_start = Instant::now();
    let mut system = ThermodynamicSystem::with_expr(particle_count, dim, 2.0, griewank_expr);
    let init_time = init_start.elapsed();
    println!("    Init time: {:.1}ms", init_time.as_secs_f64() * 1000.0);

    // Warmup
    for _ in 0..10 {
        system.step();
    }

    // Run optimization
    println!("  Running {} step optimization...", steps);
    let start = Instant::now();
    for step in 0..steps {
        let progress = step as f32 / steps as f32;
        let temp = 2.0 * (0.001_f32 / 2.0).powf(progress);
        system.set_temperature(temp);
        system.step();

        if (step + 1) % 50 == 0 {
            let particles = system.read_particles();
            let min = particles
                .iter()
                .filter(|p| !p.energy.is_nan())
                .map(|p| p.energy)
                .fold(f32::MAX, f32::min);
            println!(
                "    Step {}: min_loss = {:.4}, temp = {:.6}",
                step + 1,
                min,
                temp
            );
        }
    }
    let elapsed = start.elapsed();

    let particles = system.read_particles();
    let valid_energies: Vec<f32> = particles
        .iter()
        .filter(|p| !p.energy.is_nan() && p.energy.is_finite())
        .map(|p| p.energy)
        .collect();

    let min_energy = valid_energies.iter().cloned().fold(f32::MAX, f32::min);
    let mean_energy = valid_energies.iter().sum::<f32>() / valid_energies.len() as f32;
    let steps_per_sec = steps as f64 / elapsed.as_secs_f64();

    println!();
    println!("  Results:");
    println!("    Time: {:?}", elapsed);
    println!("    Throughput: {:.0} steps/sec", steps_per_sec);
    println!(
        "    Valid particles: {}/{}",
        valid_energies.len(),
        particle_count
    );
    println!("    Min loss: {:.4}", min_energy);
    println!("    Mean loss: {:.4}", mean_energy);
}
