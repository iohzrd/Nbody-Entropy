//! Visualization utilities for thermodynamic computing demos
//!
//! This module provides shared code for iced-based visualizations:
//! - Standard 2D optimization landscapes (Rastrigin, Rosenbrock, etc.)
//! - Drawing utilities for energy surfaces and particles
//!
//! # Example
//! ```ignore
//! use temper::viz::{Landscape, drawing};
//!
//! let landscape = Landscape::Rastrigin;
//! let energy = landscape.energy(0.0, 0.0);  // Evaluate at origin
//! let expr = landscape.expr();  // Get WGSL for GPU
//! ```

pub mod drawing;
pub mod landscapes;

pub use drawing::*;
pub use landscapes::*;
