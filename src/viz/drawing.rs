//! Common drawing utilities for iced canvas visualizations

use super::landscapes::{Landscape, X_MAX, X_MIN, Y_MAX, Y_MIN};
use iced::widget::canvas;
use iced::{Color, Point, Size};

/// Convert normalized energy [0, 1] to a color gradient
/// Dark blue (low) -> cyan -> green -> yellow -> red (high)
pub fn energy_to_color(norm: f32) -> Color {
    let (r, g, b) = if norm < 0.25 {
        let t = norm / 0.25;
        (0.0, t * 0.5, 0.3 + 0.4 * t) // dark blue -> teal
    } else if norm < 0.5 {
        let t = (norm - 0.25) / 0.25;
        (t * 0.2, 0.5 + 0.3 * t, 0.7 - 0.3 * t) // teal -> green
    } else if norm < 0.75 {
        let t = (norm - 0.5) / 0.25;
        (0.2 + 0.8 * t, 0.8 - 0.2 * t, 0.4 - 0.4 * t) // green -> orange
    } else {
        let t = (norm - 0.75) / 0.25;
        (1.0, 0.6 - 0.6 * t, 0.0) // orange -> red
    };

    Color::from_rgb(r, g, b)
}

/// Convert temperature to a color (blue = cold, red = hot)
pub fn temperature_to_color(temperature: f32, t_min: f32, t_max: f32) -> Color {
    let t_norm = ((temperature.ln() - t_min.ln()) / (t_max.ln() - t_min.ln())).clamp(0.0, 1.0);
    Color::from_rgb(
        0.3 + 0.7 * t_norm, // R: increases with temp
        0.9 - 0.6 * t_norm, // G: decreases with temp
        1.0 - 0.8 * t_norm, // B: decreases with temp
    )
}

/// Draw the energy surface heatmap for a landscape
pub fn draw_energy_surface(
    frame: &mut canvas::Frame,
    size: Size,
    landscape: &Landscape,
    resolution: usize,
) {
    draw_energy_surface_at(frame, size, 0.0, 0.0, landscape, resolution);
}

/// Draw the energy surface heatmap at an offset position
pub fn draw_energy_surface_at(
    frame: &mut canvas::Frame,
    size: Size,
    offset_x: f32,
    offset_y: f32,
    landscape: &Landscape,
    resolution: usize,
) {
    let cell_w = size.width / resolution as f32;
    let cell_h = size.height / resolution as f32;

    // Find energy range for normalization
    let mut min_e = f32::MAX;
    let mut max_e = f32::MIN;

    for iy in 0..resolution {
        for ix in 0..resolution {
            let x = X_MIN + (ix as f32 + 0.5) / resolution as f32 * (X_MAX - X_MIN);
            let y = Y_MIN + (iy as f32 + 0.5) / resolution as f32 * (Y_MAX - Y_MIN);
            let e = landscape.energy(x, y);
            if e.is_finite() {
                min_e = min_e.min(e);
                max_e = max_e.max(e);
            }
        }
    }

    // Use log scale for better visualization
    let log_min = min_e.max(0.001).ln();
    let log_max = max_e.max(0.01).ln();

    // Draw cells
    for iy in 0..resolution {
        for ix in 0..resolution {
            let x = X_MIN + (ix as f32 + 0.5) / resolution as f32 * (X_MAX - X_MIN);
            let y = Y_MIN + (iy as f32 + 0.5) / resolution as f32 * (Y_MAX - Y_MIN);
            let e = landscape.energy(x, y);

            let norm = if e.is_finite() && log_max > log_min {
                ((e.max(0.001).ln() - log_min) / (log_max - log_min)).clamp(0.0, 1.0)
            } else {
                1.0
            };

            let color = energy_to_color(norm);

            let rect = canvas::Path::rectangle(
                Point::new(offset_x + ix as f32 * cell_w, offset_y + iy as f32 * cell_h),
                Size::new(cell_w + 0.5, cell_h + 0.5),
            );
            frame.fill(&rect, color);
        }
    }
}

/// Draw axis lines at x=0 and y=0
pub fn draw_axes(frame: &mut canvas::Frame, size: Size) {
    draw_axes_at(frame, size, 0.0, 0.0);
}

/// Draw axis lines at an offset position
pub fn draw_axes_at(frame: &mut canvas::Frame, size: Size, offset_x: f32, offset_y: f32) {
    let center_x = offset_x + (-X_MIN / (X_MAX - X_MIN)) * size.width;
    let center_y = offset_y + (-Y_MIN / (Y_MAX - Y_MIN)) * size.height;

    let axis_color = Color::from_rgba(1.0, 1.0, 1.0, 0.3);

    // Vertical axis (x=0)
    if center_x > offset_x && center_x < offset_x + size.width {
        let vline = canvas::Path::line(
            Point::new(center_x, offset_y),
            Point::new(center_x, offset_y + size.height),
        );
        frame.stroke(
            &vline,
            canvas::Stroke::default()
                .with_color(axis_color)
                .with_width(1.0),
        );
    }

    // Horizontal axis (y=0)
    if center_y > offset_y && center_y < offset_y + size.height {
        let hline = canvas::Path::line(
            Point::new(offset_x, center_y),
            Point::new(offset_x + size.width, center_y),
        );
        frame.stroke(
            &hline,
            canvas::Stroke::default()
                .with_color(axis_color)
                .with_width(1.0),
        );
    }
}

/// Draw particles with glow effect
pub fn draw_particles(
    frame: &mut canvas::Frame,
    size: Size,
    xs: &[f32],
    ys: &[f32],
    color: Color,
    radius: f32,
) {
    draw_particles_at(frame, size, 0.0, 0.0, xs, ys, color, radius);
}

/// Draw particles at an offset position
pub fn draw_particles_at(
    frame: &mut canvas::Frame,
    size: Size,
    offset_x: f32,
    offset_y: f32,
    xs: &[f32],
    ys: &[f32],
    color: Color,
    radius: f32,
) {
    for (x, y) in xs.iter().zip(ys.iter()) {
        let sx = offset_x + (*x - X_MIN) / (X_MAX - X_MIN) * size.width;
        let sy = offset_y + (*y - Y_MIN) / (Y_MAX - Y_MIN) * size.height;

        if sx >= offset_x
            && sx <= offset_x + size.width
            && sy >= offset_y
            && sy <= offset_y + size.height
        {
            // Draw glow/halo
            let glow = canvas::Path::circle(Point::new(sx, sy), radius * 1.5);
            frame.fill(&glow, Color::from_rgba(1.0, 1.0, 1.0, 0.2));

            // Draw particle
            let circle = canvas::Path::circle(Point::new(sx, sy), radius);
            frame.fill(&circle, color);
        }
    }
}

/// Map world coordinates to screen coordinates
pub fn world_to_screen(x: f32, y: f32, size: Size) -> (f32, f32) {
    let sx = (x - X_MIN) / (X_MAX - X_MIN) * size.width;
    let sy = (y - Y_MIN) / (Y_MAX - Y_MIN) * size.height;
    (sx, sy)
}

/// Map screen coordinates to world coordinates
pub fn screen_to_world(sx: f32, sy: f32, size: Size) -> (f32, f32) {
    let x = X_MIN + sx / size.width * (X_MAX - X_MIN);
    let y = Y_MIN + sy / size.height * (Y_MAX - Y_MIN);
    (x, y)
}
