//! RngCore implementation using thermodynamic entropy

use super::system::ThermodynamicSystem;
use rand_core::RngCore;

/// A random number generator backed by the thermodynamic particle system.
///
/// This wraps a `ThermodynamicSystem` and implements `RngCore`, allowing
/// it to be used as a standard Rust RNG with the `rand` ecosystem.
///
/// # Example
///
/// ```ignore
/// use temper::ThermodynamicRng;
/// use rand_core::RngCore;
///
/// let mut rng = ThermodynamicRng::new(1000, 2);
///
/// // Use as standard RNG
/// let random_u32 = rng.next_u32();
/// let random_u64 = rng.next_u64();
///
/// // Fill a buffer with random bytes
/// let mut buffer = [0u8; 32];
/// rng.fill_bytes(&mut buffer);
/// ```
///
/// # Performance
///
/// The RNG batches entropy extraction from the GPU for efficiency:
/// - Steps the particle system to generate new entropy
/// - Extracts entropy in batches (particle_count * u32s per batch)
/// - Caches entropy locally to minimize GPU round-trips
///
/// For maximum throughput, use larger particle counts.
pub struct ThermodynamicRng {
    system: ThermodynamicSystem,
    buffer: Vec<u32>,
    index: usize,
}

impl ThermodynamicRng {
    /// Create a new thermodynamic RNG with default settings.
    ///
    /// Uses high temperature (T=10) for maximum entropy extraction.
    ///
    /// # Arguments
    /// * `particle_count` - Number of particles (more = more entropy per step)
    /// * `dim` - Dimensionality (2 is sufficient for entropy)
    pub fn new(particle_count: usize, dim: usize) -> Self {
        let system = ThermodynamicSystem::new(particle_count, dim, 10.0);
        Self {
            system,
            buffer: Vec::new(),
            index: 0,
        }
    }

    /// Create from an existing ThermodynamicSystem.
    ///
    /// Note: For entropy generation, temperature should be > 1.0
    pub fn from_system(system: ThermodynamicSystem) -> Self {
        Self {
            system,
            buffer: Vec::new(),
            index: 0,
        }
    }

    /// Refill the internal entropy buffer from the GPU
    fn refill(&mut self) {
        self.system.step();
        self.buffer = self.system.extract_entropy();
        self.index = 0;
    }

    /// Get the next u32 from the buffer, refilling if necessary
    fn next_from_buffer(&mut self) -> u32 {
        if self.index >= self.buffer.len() {
            self.refill();
        }
        let val = self.buffer[self.index];
        self.index += 1;
        val
    }

    /// Get access to the underlying system for configuration
    pub fn system(&self) -> &ThermodynamicSystem {
        &self.system
    }

    /// Get mutable access to the underlying system
    pub fn system_mut(&mut self) -> &mut ThermodynamicSystem {
        &mut self.system
    }

    /// Set the temperature (higher = more entropy, lower = more deterministic)
    pub fn set_temperature(&mut self, temp: f32) {
        self.system.set_temperature(temp);
    }
}

impl RngCore for ThermodynamicRng {
    fn next_u32(&mut self) -> u32 {
        self.next_from_buffer()
    }

    fn next_u64(&mut self) -> u64 {
        let lo = self.next_from_buffer() as u64;
        let hi = self.next_from_buffer() as u64;
        (hi << 32) | lo
    }

    fn fill_bytes(&mut self, dest: &mut [u8]) {
        let mut i = 0;
        while i < dest.len() {
            let val = self.next_from_buffer();
            let bytes = val.to_le_bytes();
            let remaining = dest.len() - i;
            let to_copy = remaining.min(4);
            dest[i..i + to_copy].copy_from_slice(&bytes[..to_copy]);
            i += to_copy;
        }
    }
}
