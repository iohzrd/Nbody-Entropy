//! Unified Thermodynamic Particle System
//!
//! Demonstrates that entropy generation, Bayesian sampling, and optimization
//! are all the same algorithm with different temperature settings:
//!
//! - T >> 1.0  : Entropy mode - chaotic exploration, extract random bits
//! - T ~ 0.1   : Sampling mode - SVGD/Langevin samples from posterior
//! - T → 0    : Optimize mode - gradient descent to minima
//!
//! This is the core thesis: thermodynamic computation as a unifying framework.

mod rng;
mod scheduler;
mod system;
mod types;

pub use rng::ThermodynamicRng;
pub use scheduler::AdaptiveScheduler;
pub use system::ThermodynamicSystem;
pub use types::{
    DiversityMetrics, LossFunction, ThermodynamicMode, ThermodynamicParticle, ThermodynamicStats,
};
