# Temper Binaries

This directory contains demos, visualizations, and benchmarks for the temper library.

## Visualizations (require `viz gpu` features)

Interactive iced-based visualizations:

| Binary | Description |
|--------|-------------|
| `landscape-viz` | 2D loss landscape visualizer with particles and annealing |
| `parallel-tempering-viz` | Replica exchange visualization across 4 temperature levels |
| `thermodynamic-viz` | Basic thermodynamic particle system visualization |
| `annealing-demo` | Simulated annealing demo with temperature control |
| `image-diffusion-viz` | Grayscale image reconstruction with diffusion |
| `rgb-image-diffusion-viz` | RGB image reconstruction with diffusion |

Run with:
```bash
cargo run --release --features "viz gpu" --bin <name>
```

## Function-Specific Annealing Demos (require `viz gpu`)

Focused visualizers for specific optimization landscapes:

| Binary | Description |
|--------|-------------|
| `rastrigin-annealing` | Rastrigin function (many local minima) |
| `ackley-annealing` | Ackley function (sharp global minimum) |
| `rosenbrock-annealing` | Rosenbrock banana valley |
| `schwefel-annealing` | Schwefel function (deceptive minima) |
| `adaptive-annealing` | Adaptive temperature scheduling demo |

## Benchmarks (require `gpu`)

Performance and scaling tests:

| Binary | Description |
|--------|-------------|
| `benchmark` | Core system performance |
| `thermodynamic-benchmark` | Thermodynamic simulation benchmarks |
| `classic-benchmark` | Classic optimization function benchmarks |
| `nn-benchmark` | Neural network loss landscape |
| `deep-nn-benchmark` | Deep neural network benchmarks |
| `gradient-benchmark` | Gradient computation benchmarks |
| `high-dim-benchmark` | High-dimensional scaling tests |
| `scale-test` | Particle count scaling (up to 256 dims) |
| `f16-benchmark` | f16 vs f32 precision comparison |

Run with:
```bash
cargo run --release --features gpu --bin <name>
```

## Advanced Demos (require `gpu`)

More sophisticated usage examples:

| Binary | Description |
|--------|-------------|
| `parallel-tempering` | CLI parallel tempering demo |
| `bayesian-sampling` | Bayesian posterior sampling |
| `bayesian-uncertainty` | Uncertainty quantification |
| `mode-discovery` | Multi-modal distribution exploration |
| `optimizer-comparison` | Compare thermodynamic vs classical optimizers |
| `hyperparam-tuning` | Hyperparameter optimization demo |
| `thermodynamic-stream` | Streaming thermodynamic computation |
| `thermodynamic-8d` | 8-dimensional optimization |
| `custom-expr-demo` | Custom WGSL loss function examples |
| `rng-demo` | Random number generation demo |
| `image-diffusion-demo` | CLI image diffusion |
| `rgb-image-diffusion` | RGB image diffusion (CLI) |

## Development/Testing

| Binary | Description |
|--------|-------------|
| `gpu-stress-test` | GPU stress testing |
| `high-dim-test` | High-dimensional correctness tests |
| `rosenbrock-f16-investigation` | f16 precision investigation |
| `perf-profile` | Performance profiling helper |
| `large-network` | Large network tests |
| `diffusion-demo` | Basic diffusion demo |
| `real-world-ml` | Real-world ML benchmark suite |

## Common Controls (for visualizations)

Most visualizations share these controls:
- `Space` - Pause/Resume
- `R` - Reset simulation
- `L` - Switch landscape/function
- `+/-` - Adjust simulation speed
- `T` - Toggle temperature mode (some demos)
- Arrow keys - Adjust parameters (some demos)
