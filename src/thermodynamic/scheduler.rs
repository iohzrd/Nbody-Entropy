//! Adaptive temperature scheduler for simulated annealing

/// Adaptive temperature scheduler for simulated annealing
///
/// Adjusts cooling rate based on optimization progress:
/// - Convergence detection: cool faster when near optimum
/// - Stall detection: slow down or reheat when stuck
/// - Dimension-aware: parameters scale with problem dimension
#[derive(Debug, Clone)]
pub struct AdaptiveScheduler {
    temperature: f32,
    t_start: f32,
    t_end: f32,
    base_cooling_rate: f32,
    energy_history: Vec<f32>,
    stall_count: u32,
    reheat_count: u32,
    convergence_threshold: f32,
    dim: usize,
    max_reheats: u32,
    #[allow(dead_code)]
    total_steps: u32,
}

impl AdaptiveScheduler {
    /// Create a new adaptive scheduler
    ///
    /// # Arguments
    /// * `t_start` - Starting temperature (high = more exploration)
    /// * `t_end` - Ending temperature (low = pure optimization)
    /// * `convergence_threshold` - Energy below which we consider converged
    /// * `dim` - Problem dimensionality (affects parameter scaling)
    pub fn new(t_start: f32, t_end: f32, convergence_threshold: f32, dim: usize) -> Self {
        Self::with_steps(t_start, t_end, convergence_threshold, dim, 5000)
    }

    /// Create with custom step count for cooling rate calculation
    pub fn with_steps(
        t_start: f32,
        t_end: f32,
        convergence_threshold: f32,
        dim: usize,
        total_steps: u32,
    ) -> Self {
        let base_cooling_rate = (t_end / t_start).powf(1.0 / total_steps as f32);
        let max_reheats = if dim <= 4 {
            20
        } else if dim <= 8 {
            10
        } else {
            5
        };

        Self {
            temperature: t_start,
            t_start,
            t_end,
            base_cooling_rate,
            energy_history: Vec::new(),
            stall_count: 0,
            reheat_count: 0,
            convergence_threshold,
            dim,
            max_reheats,
            total_steps,
        }
    }

    /// Update temperature based on current minimum energy
    ///
    /// Call this once per step with the best energy found so far.
    /// Returns the new temperature to use.
    pub fn update(&mut self, min_energy: f32) -> f32 {
        self.energy_history.push(min_energy);

        if min_energy < self.convergence_threshold {
            self.temperature *= self.base_cooling_rate.powf(2.0);
            self.stall_count = 0;
            self.temperature = self.temperature.max(self.t_end);
            return self.temperature;
        }

        let window = 50 + self.dim * 5;
        if self.energy_history.len() < window {
            self.temperature *= self.base_cooling_rate;
            return self.temperature;
        }

        let recent_start = self.energy_history.len() - window;
        let old_energy = self.energy_history[recent_start];
        let improvement = old_energy - min_energy;
        let improvement_rate = if old_energy > 0.01 {
            improvement / old_energy
        } else {
            improvement
        };

        let dim_factor = (self.dim as f32).sqrt() / 1.4;
        let stall_threshold = if min_energy > 100.0 {
            0.0005 / dim_factor
        } else if min_energy > 10.0 {
            0.002 / dim_factor
        } else if min_energy > 1.0 {
            0.005 / dim_factor
        } else {
            0.02 / dim_factor
        };

        let stall_required = 80 + self.dim as u32 * 10;

        if improvement_rate < stall_threshold && self.temperature > self.t_end * 10.0 {
            self.stall_count += 1;
            if self.stall_count > stall_required
                && min_energy > self.convergence_threshold * 10.0
                && self.reheat_count < self.max_reheats
            {
                let reheat_factor = if self.dim <= 4 { 2.0 } else { 1.3 };
                self.temperature = (self.temperature * reheat_factor).min(self.t_start * 0.3);
                self.stall_count = 0;
                self.reheat_count += 1;
            } else {
                self.temperature *= self.base_cooling_rate.powf(0.5);
            }
        } else if improvement_rate > stall_threshold * 3.0 {
            self.temperature *= self.base_cooling_rate.powf(1.5);
            self.stall_count = 0;
        } else {
            self.temperature *= self.base_cooling_rate;
            self.stall_count = self.stall_count.saturating_sub(1);
        }

        self.temperature = self.temperature.max(self.t_end);
        self.temperature
    }

    /// Reset the scheduler to initial state
    pub fn reset(&mut self) {
        self.temperature = self.t_start;
        self.energy_history.clear();
        self.stall_count = 0;
        self.reheat_count = 0;
    }

    /// Get current temperature
    pub fn temperature(&self) -> f32 {
        self.temperature
    }

    /// Get number of reheats performed
    pub fn reheat_count(&self) -> u32 {
        self.reheat_count
    }

    /// Get the energy history
    pub fn energy_history(&self) -> &[f32] {
        &self.energy_history
    }

    /// Check if scheduler detected convergence
    pub fn is_converged(&self) -> bool {
        self.energy_history
            .last()
            .map_or(false, |&e| e < self.convergence_threshold)
    }

    /// Get progress as fraction (0.0 to 1.0) based on temperature
    pub fn progress(&self) -> f32 {
        let log_t =
            (self.temperature.ln() - self.t_end.ln()) / (self.t_start.ln() - self.t_end.ln());
        1.0 - log_t.clamp(0.0, 1.0)
    }
}
