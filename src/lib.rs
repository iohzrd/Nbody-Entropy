pub mod gpu;
pub use gpu::GpuNbodyEntropy;

pub const DEFAULT_PARTICLE_COUNT: usize = 64;
pub const ATTRACTOR_COUNT: usize = 8;
pub const FOLLOWER_COUNT: usize = DEFAULT_PARTICLE_COUNT - ATTRACTOR_COUNT;
