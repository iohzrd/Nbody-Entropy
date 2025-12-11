//! RngCore Demo
//!
//! Demonstrates using ThermodynamicRng as a standard Rust RNG.
//!
//! Run with: cargo run --release --features gpu --bin rng-demo

use temper::{RngCore, ThermodynamicRng};

fn main() {
    println!("ThermodynamicRng Demo");
    println!("=====================\n");
    println!("Using GPU-accelerated particle dynamics as an RNG source.\n");

    // Create a thermodynamic RNG with 1000 particles
    let mut rng = ThermodynamicRng::new(1000, 2);

    // Generate some random u32s
    println!("Random u32 values:");
    for i in 0..5 {
        println!("  [{i}] {}", rng.next_u32());
    }

    // Generate some random u64s
    println!("\nRandom u64 values:");
    for i in 0..5 {
        println!("  [{i}] {}", rng.next_u64());
    }

    // Fill a buffer with random bytes
    let mut buffer = [0u8; 32];
    rng.fill_bytes(&mut buffer);
    println!("\nRandom bytes (32):");
    print!("  ");
    for (i, byte) in buffer.iter().enumerate() {
        print!("{:02x}", byte);
        if (i + 1) % 8 == 0 {
            print!(" ");
        }
    }
    println!();

    // Generate random floats in [0, 1)
    println!("\nRandom f64 in [0, 1):");
    for i in 0..5 {
        let val = rng.next_u64() as f64 / u64::MAX as f64;
        println!("  [{i}] {:.10}", val);
    }

    // Demonstrate throughput
    println!("\nThroughput test (generating 1M random u32s)...");
    let start = std::time::Instant::now();
    let count = 1_000_000;
    let mut sum: u64 = 0;
    for _ in 0..count {
        sum = sum.wrapping_add(rng.next_u32() as u64);
    }
    let elapsed = start.elapsed();
    let rate = count as f64 / elapsed.as_secs_f64();
    println!("  Generated {} values in {:.2?}", count, elapsed);
    println!("  Rate: {:.2} values/sec", rate);
    println!("  Sum (to prevent optimization): {}", sum);

    println!("\nUsage in your code:");
    println!("  use temper::{{ThermodynamicRng, RngCore}};");
    println!("  let mut rng = ThermodynamicRng::new(1000, 2);");
    println!("  let random_value = rng.next_u32();");
}
