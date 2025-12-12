//! The core ThermodynamicSystem GPU particle simulation

use super::types::{
    DiversityMetrics, LossFunction, MAX_DIMENSIONS, MAX_PARTICLES, ThermodynamicMode,
    ThermodynamicParticle, ThermodynamicStats, Uniforms,
};
use half::f16;
use wgpu::util::DeviceExt;

/// Get random seed from OS entropy via getrandom
fn random_seed() -> u64 {
    let mut buf = [0u8; 8];
    getrandom::fill(&mut buf).expect("Failed to get OS entropy");
    u64::from_le_bytes(buf)
}

// F16 position update code for shader specialization
const F16_POSITION_UPDATE: &str = r#"
        // F16 COMPUTE PATH - all position arithmetic in f16
        let grad_f16 = f16(grad_clipped);
        let gamma_f16 = f16(uniforms.gamma);
        let dt_f16 = f16(uniforms.dt);
        let rep_scale_f16 = f16(repulsion_scale);
        let noise_scale_f16 = f16(noise_scale);
        let noise_f16 = f16(noise);

        // Update directly in f16
        let grad_term_f16 = -gamma_f16 * grad_f16;
        let repulsion_term_f16 = repulsion[idx].pos[d] * rep_scale_f16;
        let noise_term_f16 = noise_scale_f16 * noise_f16;

        var new_pos_f16 = p.pos[d] + (grad_term_f16 + repulsion_term_f16) * dt_f16 + noise_term_f16;

        // Clamp in f16
        if uniforms.loss_fn == 9u {
            new_pos_f16 = clamp(new_pos_f16, f16(-500.0), f16(500.0));
        } else {
            new_pos_f16 = clamp(new_pos_f16, f16(-5.0), f16(5.0));
        }

        p.pos[d] = new_pos_f16;
        pos_f32[d] = f32(new_pos_f16); // For entropy extraction
"#;

/// Generate shader with f16 or f32 position update code (compile-time specialization)
fn specialize_shader(base_shader: &str, use_f16: bool) -> String {
    if !use_f16 {
        return base_shader.to_string();
    }

    let start_marker = "// POSITION_UPDATE_CODE_START";
    let end_marker = "// POSITION_UPDATE_CODE_END";

    if let Some(start_idx) = base_shader.find(start_marker) {
        if let Some(end_idx) = base_shader.find(end_marker) {
            let before = &base_shader[..start_idx];
            let after = &base_shader[end_idx + end_marker.len()..];
            return format!(
                "{}// F16 COMPUTE (compile-time specialized)\n{}{}",
                before, F16_POSITION_UPDATE, after
            );
        }
    }

    base_shader.to_string()
}

/// Unified thermodynamic particle system
///
/// Demonstrates that entropy generation, Bayesian sampling, and optimization
/// are all the same algorithm with different temperature settings:
///
/// - T >> 1.0  : Entropy mode - chaotic exploration, extract random bits
/// - T ~ 0.1   : Sampling mode - SVGD/Langevin samples from posterior
/// - T → 0    : Optimize mode - gradient descent to minima
pub struct ThermodynamicSystem {
    device: wgpu::Device,
    queue: wgpu::Queue,
    particle_buffer: wgpu::Buffer,
    #[allow(dead_code)]
    repulsion_buffer: wgpu::Buffer,
    entropy_buffer: wgpu::Buffer,
    staging_buffer: wgpu::Buffer,
    entropy_staging: wgpu::Buffer,
    uniform_buffer: wgpu::Buffer,
    repulsion_pipeline: wgpu::ComputePipeline,
    update_pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    particle_count: usize,
    dim: usize,
    temperature: f32,
    gamma: f32,
    repulsion_strength: f32,
    kernel_bandwidth: f32,
    dt: f32,
    step: u32,
    loss_fn: LossFunction,
    repulsion_samples: u32,
    use_f16_compute: bool,
    base_seed: u32,
    #[allow(dead_code)]
    entropy_pool: Vec<u32>,
    #[allow(dead_code)]
    custom_loss_wgsl: Option<String>,
}

impl ThermodynamicSystem {
    pub fn new(particle_count: usize, dim: usize, temperature: f32) -> Self {
        assert!(particle_count <= MAX_PARTICLES);
        assert!(dim <= MAX_DIMENSIONS);

        let mut particles = vec![ThermodynamicParticle::default(); particle_count];
        let base_seed = random_seed();

        let mut seed = base_seed;
        for p in particles.iter_mut() {
            for d in 0..dim {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                let val = -4.0 + (seed & 0xFFFF) as f32 / 65535.0 * 8.0;
                p.pos[d] = f16::from_f32(val);
            }
        }

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))
        .expect("No GPU adapter found");

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            required_features: wgpu::Features::SHADER_F16,
            ..Default::default()
        }))
        .expect("Failed to create device with f16 support");

        let particle_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("particles"),
            contents: bytemuck::cast_slice(&particles),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

        let repulsion_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("repulsion"),
            size: (particle_count * std::mem::size_of::<ThermodynamicParticle>()) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let entropy_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("entropy"),
            size: (particle_count * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: (particle_count * std::mem::size_of::<ThermodynamicParticle>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let entropy_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("entropy_staging"),
            size: (particle_count * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let (gamma, repulsion_strength, kernel_bandwidth, dt) =
            Self::params_for_temperature(temperature, dim);
        let repulsion_samples = if temperature < 0.01 { 0 } else { 64 };

        let uniforms = Uniforms {
            particle_count: particle_count as u32,
            dim: dim as u32,
            gamma,
            temperature,
            repulsion_strength,
            kernel_bandwidth,
            dt,
            seed: base_seed as u32,
            mode: ThermodynamicMode::from_temperature(temperature) as u32,
            loss_fn: LossFunction::default() as u32,
            repulsion_samples,
            use_f16_compute: 0,
        };

        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("uniforms"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let base_shader = include_str!("../shaders/thermodynamic.wgsl");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("thermodynamic"),
            source: wgpu::ShaderSource::Wgsl(base_shader.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("thermodynamic_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("thermodynamic_bind_group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: particle_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: repulsion_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: entropy_buffer.as_entire_binding(),
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("thermodynamic_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let repulsion_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("repulsion_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("compute_repulsion"),
            compilation_options: Default::default(),
            cache: None,
        });

        let update_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("update_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("update_particles"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            device,
            queue,
            particle_buffer,
            repulsion_buffer,
            entropy_buffer,
            staging_buffer,
            entropy_staging,
            uniform_buffer,
            repulsion_pipeline,
            update_pipeline,
            bind_group,
            particle_count,
            dim,
            temperature,
            gamma,
            repulsion_strength,
            kernel_bandwidth,
            dt,
            step: 0,
            loss_fn: LossFunction::default(),
            repulsion_samples,
            use_f16_compute: false,
            base_seed: base_seed as u32,
            entropy_pool: Vec::new(),
            custom_loss_wgsl: None,
        }
    }

    /// Create with a specific loss function
    pub fn with_loss_function(
        particle_count: usize,
        dim: usize,
        temperature: f32,
        loss_fn: LossFunction,
    ) -> Self {
        let mut sys = Self::new(particle_count, dim, temperature);
        sys.set_loss_function(loss_fn);
        sys
    }

    /// Create with f16 compute enabled (compile-time shader specialization)
    pub fn with_f16_compute(
        particle_count: usize,
        dim: usize,
        temperature: f32,
        loss_fn: LossFunction,
    ) -> Self {
        Self::new_internal(particle_count, dim, temperature, loss_fn, true)
    }

    /// Internal constructor with all options
    fn new_internal(
        particle_count: usize,
        dim: usize,
        temperature: f32,
        loss_fn: LossFunction,
        use_f16_compute: bool,
    ) -> Self {
        assert!(particle_count <= MAX_PARTICLES);
        assert!(dim <= MAX_DIMENSIONS);

        let mut particles = vec![ThermodynamicParticle::default(); particle_count];
        let base_seed = random_seed();

        let mut seed = base_seed;
        for p in particles.iter_mut() {
            for d in 0..dim {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                let val = -4.0 + (seed & 0xFFFF) as f32 / 65535.0 * 8.0;
                p.pos[d] = f16::from_f32(val);
            }
        }

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))
        .expect("No GPU adapter found");

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            required_features: wgpu::Features::SHADER_F16,
            ..Default::default()
        }))
        .expect("Failed to create device with f16 support");

        let particle_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("particles"),
            contents: bytemuck::cast_slice(&particles),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

        let repulsion_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("repulsion"),
            size: (particle_count * std::mem::size_of::<ThermodynamicParticle>()) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let entropy_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("entropy"),
            size: (particle_count * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: (particle_count * std::mem::size_of::<ThermodynamicParticle>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let entropy_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("entropy_staging"),
            size: (particle_count * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let (gamma, repulsion_strength, kernel_bandwidth, dt) =
            Self::params_for_temperature(temperature, dim);
        let repulsion_samples = if temperature < 0.01 { 0 } else { 64 };

        let uniforms = Uniforms {
            particle_count: particle_count as u32,
            dim: dim as u32,
            gamma,
            temperature,
            repulsion_strength,
            kernel_bandwidth,
            dt,
            seed: base_seed as u32,
            mode: ThermodynamicMode::from_temperature(temperature) as u32,
            loss_fn: loss_fn as u32,
            repulsion_samples,
            use_f16_compute: if use_f16_compute { 1 } else { 0 },
        };

        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("uniforms"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let base_shader = include_str!("../shaders/thermodynamic.wgsl");
        let specialized_shader = specialize_shader(base_shader, use_f16_compute);
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(if use_f16_compute {
                "thermodynamic_f16"
            } else {
                "thermodynamic_f32"
            }),
            source: wgpu::ShaderSource::Wgsl(specialized_shader.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("thermodynamic_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("thermodynamic_bind_group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: particle_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: repulsion_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: entropy_buffer.as_entire_binding(),
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("thermodynamic_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let repulsion_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("repulsion_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("compute_repulsion"),
            compilation_options: Default::default(),
            cache: None,
        });

        let update_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("update_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("update_particles"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            device,
            queue,
            particle_buffer,
            repulsion_buffer,
            entropy_buffer,
            staging_buffer,
            entropy_staging,
            uniform_buffer,
            repulsion_pipeline,
            update_pipeline,
            bind_group,
            particle_count,
            dim,
            temperature,
            gamma,
            repulsion_strength,
            kernel_bandwidth,
            dt,
            step: 0,
            loss_fn,
            repulsion_samples,
            use_f16_compute,
            base_seed: base_seed as u32,
            entropy_pool: Vec::new(),
            custom_loss_wgsl: None,
        }
    }

    /// Create with a custom expression-based loss function
    pub fn with_expr(
        particle_count: usize,
        dim: usize,
        temperature: f32,
        expr: crate::expr::Expr,
    ) -> Self {
        Self::with_expr_options(particle_count, dim, temperature, expr, true)
    }

    /// Create with a custom expression and control gradient type
    pub fn with_expr_options(
        particle_count: usize,
        dim: usize,
        temperature: f32,
        expr: crate::expr::Expr,
        analytical_gradients: bool,
    ) -> Self {
        assert!(particle_count <= MAX_PARTICLES);
        assert!(dim <= MAX_DIMENSIONS);

        let mut particles = vec![ThermodynamicParticle::default(); particle_count];
        let base_seed = random_seed();

        let mut seed = base_seed;
        for p in particles.iter_mut() {
            for d in 0..dim {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                let val = -4.0 + (seed & 0xFFFF) as f32 / 65535.0 * 8.0;
                p.pos[d] = f16::from_f32(val);
            }
        }

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))
        .expect("No GPU adapter found");

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            required_features: wgpu::Features::SHADER_F16,
            ..Default::default()
        }))
        .expect("Failed to create device with f16 support");

        let particle_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("particles"),
            contents: bytemuck::cast_slice(&particles),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

        let repulsion_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("repulsion"),
            size: (particle_count * std::mem::size_of::<ThermodynamicParticle>()) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let entropy_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("entropy"),
            size: (particle_count * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: (particle_count * std::mem::size_of::<ThermodynamicParticle>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let entropy_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("entropy_staging"),
            size: (particle_count * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let (gamma, repulsion_strength, kernel_bandwidth, dt) =
            Self::params_for_temperature(temperature, dim);
        let repulsion_samples = if temperature < 0.01 { 0 } else { 64 };

        let uniforms = Uniforms {
            particle_count: particle_count as u32,
            dim: dim as u32,
            gamma,
            temperature,
            repulsion_strength,
            kernel_bandwidth,
            dt,
            seed: base_seed as u32,
            mode: ThermodynamicMode::from_temperature(temperature) as u32,
            loss_fn: LossFunction::Custom as u32,
            repulsion_samples,
            use_f16_compute: 0,
        };

        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("uniforms"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let custom_wgsl = expr.to_wgsl_with_options(analytical_gradients);
        let base_shader = include_str!("../shaders/thermodynamic.wgsl");

        let stub_marker = "// Stub functions for custom expressions";
        let stub_end = "// 2D neural net:";

        let full_shader = if let Some(stub_start) = base_shader.find(stub_marker) {
            if let Some(stub_end_pos) = base_shader[stub_start..].find(stub_end) {
                let before = &base_shader[..stub_start];
                let after = &base_shader[stub_start + stub_end_pos..];
                format!("{}{}\n\n{}", before, custom_wgsl, after)
            } else {
                format!("{}\n\n{}", custom_wgsl, base_shader)
            }
        } else {
            format!("{}\n\n{}", custom_wgsl, base_shader)
        };

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("thermodynamic_custom"),
            source: wgpu::ShaderSource::Wgsl(full_shader.clone().into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("thermodynamic_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("thermodynamic_bind_group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: particle_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: repulsion_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: entropy_buffer.as_entire_binding(),
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("thermodynamic_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let repulsion_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("repulsion_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("compute_repulsion"),
            compilation_options: Default::default(),
            cache: None,
        });

        let update_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("update_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("update_particles"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            device,
            queue,
            particle_buffer,
            repulsion_buffer,
            entropy_buffer,
            staging_buffer,
            entropy_staging,
            uniform_buffer,
            repulsion_pipeline,
            update_pipeline,
            bind_group,
            particle_count,
            dim,
            temperature,
            gamma,
            repulsion_strength,
            kernel_bandwidth,
            dt,
            step: 0,
            loss_fn: LossFunction::Custom,
            repulsion_samples,
            use_f16_compute: false,
            base_seed: base_seed as u32,
            entropy_pool: Vec::new(),
            custom_loss_wgsl: Some(full_shader),
        }
    }

    /// Get adaptive parameters for a given temperature
    fn params_for_temperature(temperature: f32, dim: usize) -> (f32, f32, f32, f32) {
        let mode = ThermodynamicMode::from_temperature(temperature);
        match mode {
            ThermodynamicMode::Optimize => (1.0, 0.0, 0.5, 0.1),
            ThermodynamicMode::Sample => {
                let bandwidth = 0.3 + 0.1 * dim as f32;
                (0.5, 0.2, bandwidth, 0.01)
            }
            ThermodynamicMode::Entropy => (0.1, 0.05, 1.0, 0.05),
        }
    }

    /// Set temperature and adapt parameters
    pub fn set_temperature(&mut self, temperature: f32) {
        self.temperature = temperature;
        let (gamma, repulsion_strength, kernel_bandwidth, dt) =
            Self::params_for_temperature(temperature, self.dim);
        self.gamma = gamma;
        self.repulsion_strength = repulsion_strength;
        self.kernel_bandwidth = kernel_bandwidth;
        self.dt = dt;
        self.repulsion_samples = if temperature < 0.01 { 0 } else { 64 };
    }

    /// Set the number of repulsion samples
    pub fn set_repulsion_samples(&mut self, samples: u32) {
        self.repulsion_samples = samples;
    }

    /// Get current repulsion samples setting
    pub fn repulsion_samples(&self) -> u32 {
        self.repulsion_samples
    }

    /// Check if f16 compute is enabled
    pub fn f16_compute(&self) -> bool {
        self.use_f16_compute
    }

    /// Set the time step (dt)
    pub fn set_dt(&mut self, dt: f32) {
        self.dt = dt;
    }

    /// Get current time step
    pub fn dt(&self) -> f32 {
        self.dt
    }

    /// Set the loss function
    pub fn set_loss_function(&mut self, loss_fn: LossFunction) {
        self.loss_fn = loss_fn;
    }

    /// Get current loss function
    pub fn loss_function(&self) -> LossFunction {
        self.loss_fn
    }

    /// Get current operating mode
    pub fn mode(&self) -> ThermodynamicMode {
        ThermodynamicMode::from_temperature(self.temperature)
    }

    /// Run one simulation step
    pub fn step(&mut self) {
        self.step += 1;

        let uniforms = Uniforms {
            particle_count: self.particle_count as u32,
            dim: self.dim as u32,
            gamma: self.gamma,
            temperature: self.temperature,
            repulsion_strength: self.repulsion_strength,
            kernel_bandwidth: self.kernel_bandwidth,
            dt: self.dt,
            seed: self
                .base_seed
                .wrapping_add(self.step.wrapping_mul(2654435769)),
            mode: self.mode() as u32,
            loss_fn: self.loss_fn as u32,
            repulsion_samples: self.repulsion_samples,
            use_f16_compute: if self.use_f16_compute { 1 } else { 0 },
        };
        self.queue
            .write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("thermodynamic_encoder"),
            });

        let workgroups = (self.particle_count as u32 + 63) / 64;

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("repulsion_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.repulsion_pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("update_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.update_pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        self.queue.submit(Some(encoder.finish()));
    }

    /// Read particles back from GPU
    pub fn read_particles(&self) -> Vec<ThermodynamicParticle> {
        let size = (self.particle_count * std::mem::size_of::<ThermodynamicParticle>()) as u64;

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("read_encoder"),
            });
        encoder.copy_buffer_to_buffer(&self.particle_buffer, 0, &self.staging_buffer, 0, size);
        self.queue.submit(Some(encoder.finish()));

        let slice = self.staging_buffer.slice(..size);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            tx.send(r).unwrap();
        });
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .unwrap();
        rx.recv().unwrap().expect("map failed");

        let data = slice.get_mapped_range();
        let particles: Vec<ThermodynamicParticle> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        self.staging_buffer.unmap();

        particles
    }

    /// Extract entropy (only valid in entropy mode)
    pub fn extract_entropy(&mut self) -> Vec<u32> {
        if self.mode() != ThermodynamicMode::Entropy {
            return Vec::new();
        }

        let size = (self.particle_count * 4) as u64;

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("entropy_encoder"),
            });
        encoder.copy_buffer_to_buffer(&self.entropy_buffer, 0, &self.entropy_staging, 0, size);
        self.queue.submit(Some(encoder.finish()));

        let slice = self.entropy_staging.slice(..size);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            tx.send(r).unwrap();
        });
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .unwrap();
        rx.recv().unwrap().expect("map failed");

        let data = slice.get_mapped_range();
        let entropy: Vec<u32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        self.entropy_staging.unmap();

        entropy
    }

    /// Compute statistics about current particle distribution
    pub fn statistics(&self) -> ThermodynamicStats {
        let particles = self.read_particles();

        let energies: Vec<f32> = particles.iter().map(|p| p.energy).collect();
        let mean_energy = energies.iter().sum::<f32>() / particles.len() as f32;
        let min_energy = energies.iter().cloned().fold(f32::MAX, f32::min);
        let max_energy = energies.iter().cloned().fold(f32::MIN, f32::max);

        let mut spread = 0.0;
        let mut count = 0;
        for i in 0..particles.len().min(100) {
            for j in (i + 1)..particles.len().min(100) {
                let dx = particles[i].pos[0].to_f32() - particles[j].pos[0].to_f32();
                let dy = particles[i].pos[1].to_f32() - particles[j].pos[1].to_f32();
                spread += (dx * dx + dy * dy).sqrt();
                count += 1;
            }
        }
        spread /= count.max(1) as f32;

        let low_energy_count = energies.iter().filter(|&&e| e < 0.1).count();

        ThermodynamicStats {
            mean_energy,
            min_energy,
            max_energy,
            spread,
            low_energy_fraction: low_energy_count as f32 / particles.len() as f32,
            mode: self.mode(),
            temperature: self.temperature,
        }
    }

    pub fn temperature(&self) -> f32 {
        self.temperature
    }

    pub fn particle_count(&self) -> usize {
        self.particle_count
    }

    /// Compute comprehensive diversity metrics for the current population
    pub fn diversity_metrics(&self) -> DiversityMetrics {
        let particles = self.read_particles();
        let n = particles.len();
        let dim = self.dim;

        let sample_size = n.min(500);
        let step = if n > sample_size { n / sample_size } else { 1 };
        let sampled: Vec<_> = particles.iter().step_by(step).take(sample_size).collect();
        let m = sampled.len();

        let mut distances = Vec::new();
        for i in 0..m {
            for j in (i + 1)..m {
                let mut dist_sq = 0.0f32;
                for d in 0..dim {
                    let diff = sampled[i].pos[d].to_f32() - sampled[j].pos[d].to_f32();
                    dist_sq += diff * diff;
                }
                distances.push(dist_sq.sqrt());
            }
        }

        let mean_dist = if !distances.is_empty() {
            distances.iter().sum::<f32>() / distances.len() as f32
        } else {
            0.0
        };

        let dist_var = if !distances.is_empty() {
            distances
                .iter()
                .map(|d| (d - mean_dist).powi(2))
                .sum::<f32>()
                / distances.len() as f32
        } else {
            0.0
        };

        let energies: Vec<f32> = particles
            .iter()
            .filter(|p| !p.energy.is_nan() && p.energy.is_finite())
            .map(|p| p.energy)
            .collect();

        let mean_energy = energies.iter().sum::<f32>() / energies.len().max(1) as f32;
        let energy_var = energies
            .iter()
            .map(|e| (e - mean_energy).powi(2))
            .sum::<f32>()
            / energies.len().max(1) as f32;

        let min_energy = energies.iter().cloned().fold(f32::MAX, f32::min);
        let weights: Vec<f32> = energies
            .iter()
            .map(|e| (-(e - min_energy) / self.temperature.max(0.001)).exp())
            .collect();
        let sum_w: f32 = weights.iter().sum();
        let sum_w_sq: f32 = weights.iter().map(|w| w * w).sum();
        let ess = if sum_w_sq > 0.0 {
            (sum_w * sum_w / sum_w_sq).min(n as f32)
        } else {
            1.0
        };

        let energy_threshold = min_energy + energy_var.sqrt();
        let low_energy: Vec<_> = sampled
            .iter()
            .filter(|p| !p.energy.is_nan() && p.energy < energy_threshold)
            .collect();

        let mut modes = 0usize;
        let mode_dist_threshold = mean_dist * 0.3;
        let mut mode_centers: Vec<Vec<f32>> = Vec::new();

        for p in &low_energy {
            let pos: Vec<f32> = (0..dim).map(|d| p.pos[d].to_f32()).collect();
            let is_new_mode = mode_centers.iter().all(|center| {
                let dist: f32 = pos
                    .iter()
                    .zip(center.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f32>()
                    .sqrt();
                dist > mode_dist_threshold
            });
            if is_new_mode {
                modes += 1;
                mode_centers.push(pos);
            }
        }

        let mut min_pos = vec![f32::MAX; dim];
        let mut max_pos = vec![f32::MIN; dim];
        for p in &sampled {
            for d in 0..dim {
                let v = p.pos[d].to_f32();
                min_pos[d] = min_pos[d].min(v);
                max_pos[d] = max_pos[d].max(v);
            }
        }

        let expected_range = 8.0f32;
        let mut coverage = 1.0f32;
        for d in 0..dim.min(8) {
            let range = (max_pos[d] - min_pos[d]).max(0.001);
            coverage *= (range / expected_range).min(1.0);
        }
        coverage = coverage.powf(1.0 / dim.min(8) as f32);

        DiversityMetrics {
            mean_pairwise_distance: mean_dist,
            distance_std: dist_var.sqrt(),
            energy_variance: energy_var,
            effective_sample_size: ess,
            estimated_modes: modes.max(1),
            coverage,
            dim,
        }
    }
}
