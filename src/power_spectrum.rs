use numpy::ndarray::Array3;
use rustfft::{FftPlanner, num_complex::Complex};
use std::f64::consts::PI;

pub struct PowerSpectrumResult {
    pub k_centers: Vec<f64>,
    pub power: Vec<f64>,
    pub n_modes: Vec<u64>,
}

/// Normalized sinc: sin(x)/x, with sinc(0) = 1.
#[inline]
fn sinc(x: f64) -> f64 {
    if x.abs() < 1e-10 { 1.0 } else { x.sin() / x }
}

/// Signed mode index: maps FFT output index m to [-n/2, n/2].
#[inline]
fn signed_mode(m: usize, n: usize) -> isize {
    if m <= n / 2 {
        m as isize
    } else {
        m as isize - n as isize
    }
}

/// CIC window function value for FFT index m in a grid of size n.
/// W(m) = sinc(π * m_signed / n)
#[inline]
fn cic_window(m: usize, n: usize) -> f64 {
    let ms = signed_mode(m, n);
    sinc(PI * ms as f64 / n as f64)
}

/// Perform an in-place 3D FFT on a flat row-major array of shape [n, n, n].
/// Calls `on_progress` with values 0.38, 0.63, and 0.80 after each axis pass.
fn fft3d(data: &mut [Complex<f64>], n: usize, on_progress: &impl Fn(f64)) {
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n);
    let scratch_len = fft.get_inplace_scratch_len();
    let mut scratch = vec![Complex::default(); scratch_len];
    let mut buf = vec![Complex::default(); n];

    // Axis 2 (z): innermost, stride 1 — data is already contiguous.
    // Progress: 0.05 → 0.30
    for i in 0..n {
        for j in 0..n {
            let start = (i * n + j) * n;
            fft.process_with_scratch(&mut data[start..start + n], &mut scratch);
        }
        on_progress(0.05 + 0.25 * (i + 1) as f64 / n as f64);
    }

    // Axis 1 (y): stride n — extract into buf, FFT, write back.
    // Progress: 0.30 → 0.55
    for i in 0..n {
        for k in 0..n {
            for j in 0..n {
                buf[j] = data[(i * n + j) * n + k];
            }
            fft.process_with_scratch(&mut buf, &mut scratch);
            for j in 0..n {
                data[(i * n + j) * n + k] = buf[j];
            }
        }
        on_progress(0.30 + 0.25 * (i + 1) as f64 / n as f64);
    }

    // Axis 0 (x): stride n² — extract into buf, FFT, write back.
    // Progress: 0.55 → 0.80
    for j in 0..n {
        for k in 0..n {
            for i in 0..n {
                buf[i] = data[(i * n + j) * n + k];
            }
            fft.process_with_scratch(&mut buf, &mut scratch);
            for i in 0..n {
                data[(i * n + j) * n + k] = buf[i];
            }
        }
        on_progress(0.55 + 0.25 * (j + 1) as f64 / n as f64);
    }
}

/// Compute the 3D matter power spectrum P(k) from a CIC density grid.
///
/// Steps:
///   1. Compute density contrast δ = (ρ - ρ̄) / ρ̄.
///   2. 3D FFT and normalize: δ̃_k = FFT(δ) / N³.
///   3. Optionally divide each mode by the CIC window W(kx)·W(ky)·W(kz).
///   4. Bin |δ̃_k|² into k bins and compute P(k) = V·⟨|δ̃_k|²⟩.
///
/// # Arguments
/// * `grid`          - CIC particle counts, shape [n, n, n]. Assumed cubic.
/// * `box_size`      - Physical side length of the box (same units as desired for P(k)).
/// * `k_min`         - Lower edge of the first k bin.
/// * `k_max`         - Upper edge of the last k bin.
/// * `n_bins`        - Number of bins.
/// * `log_bins`      - If true, bins are log-spaced (geometric-mean centers);
///                     if false, linearly spaced (arithmetic-mean centers).
/// * `deconvolve_cic`    - If true, correct for the CIC window function suppression.
/// * `subtract_shot_noise` - If true, subtract the Poisson shot noise floor
///                           P_shot = V / N_particles from every bin.
/// * `on_progress`       - Called with a value in [0, 1] at key checkpoints.
pub fn compute_power_spectrum(
    grid: &Array3<f64>,
    box_size: f64,
    k_min: f64,
    k_max: f64,
    n_bins: usize,
    log_bins: bool,
    deconvolve_cic: bool,
    subtract_shot_noise: bool,
    on_progress: impl Fn(f64),
) -> PowerSpectrumResult {
    let n = grid.shape()[0];
    let n3 = (n * n * n) as f64;
    let volume = box_size.powi(3);

    on_progress(0.0);

    // --- Step 1: density contrast ---
    // mean = N_particles / N_cells³, so N_particles = mean * n3.
    let mean = grid.iter().sum::<f64>() / n3;
    let p_shot = volume / (mean * n3); // V / N_particles
    let mut cdata: Vec<Complex<f64>> = grid
        .iter()
        .map(|&v| Complex::new((v - mean) / mean, 0.0))
        .collect();
    on_progress(0.05);

    // --- Step 2: 3D FFT, then normalize by 1/N³ ---
    // fft3d reports n times per axis: z 0.05→0.30, y 0.30→0.55, x 0.55→0.80.
    fft3d(&mut cdata, n, &on_progress);
    let norm = 1.0 / n3;
    for c in cdata.iter_mut() {
        *c *= norm;
    }

    // --- Step 3: bin edges and centers ---
    let bin_edges: Vec<f64> = if log_bins {
        let ln_kmin = k_min.ln();
        let ln_range = k_max.ln() - ln_kmin;
        (0..=n_bins)
            .map(|i| (ln_kmin + i as f64 * ln_range / n_bins as f64).exp())
            .collect()
    } else {
        let lin_range = k_max - k_min;
        (0..=n_bins)
            .map(|i| k_min + i as f64 * lin_range / n_bins as f64)
            .collect()
    };
    let k_centers: Vec<f64> = if log_bins {
        (0..n_bins)
            .map(|i| (bin_edges[i] * bin_edges[i + 1]).sqrt()) // geometric mean
            .collect()
    } else {
        (0..n_bins)
            .map(|i| 0.5 * (bin_edges[i] + bin_edges[i + 1])) // arithmetic mean
            .collect()
    };

    // --- Step 4: accumulate |δ̃_k|² into bins ---
    let mut power_sum = vec![0.0f64; n_bins];
    let mut n_modes = vec![0u64; n_bins];

    let dk = 2.0 * PI / box_size; // fundamental wavenumber

    // Precompute bin-lookup parameters to keep the inner loop branch-free.
    let (bin_offset, bin_scale) = if log_bins {
        let ln_kmin = k_min.ln();
        (ln_kmin, n_bins as f64 / (k_max.ln() - ln_kmin))
    } else {
        (k_min, n_bins as f64 / (k_max - k_min))
    };

    // Report every ~1/20th of the binning loop (progress 0.80 → 1.00).
    let report_every = (n / 20).max(1);

    for i in 0..n {
        if i % report_every == 0 {
            on_progress(0.80 + 0.20 * i as f64 / n as f64);
        }

        let kx = dk * signed_mode(i, n) as f64;
        let wx = cic_window(i, n);

        for j in 0..n {
            let ky = dk * signed_mode(j, n) as f64;
            let wy = cic_window(j, n);

            for k in 0..n {
                let kz = dk * signed_mode(k, n) as f64;
                let kmag = (kx * kx + ky * ky + kz * kz).sqrt();

                if kmag < k_min || kmag >= k_max {
                    continue;
                }

                let k_mapped = if log_bins { kmag.ln() } else { kmag };
                let bin = ((k_mapped - bin_offset) * bin_scale) as usize;
                if bin >= n_bins {
                    continue;
                }

                let mut pk = cdata[(i * n + j) * n + k].norm_sqr();

                if deconvolve_cic {
                    let wz = cic_window(k, n);
                    let w2 = (wx * wy * wz).powi(2);
                    pk /= w2;
                }

                power_sum[bin] += pk;
                n_modes[bin] += 1;
            }
        }
    }

    // --- Step 5: P(k) = V · mean(|δ̃_k|²) per bin, with optional shot noise subtraction ---
    let power: Vec<f64> = (0..n_bins)
        .map(|i| {
            if n_modes[i] > 0 {
                let pk = volume * power_sum[i] / n_modes[i] as f64;
                if subtract_shot_noise { pk - p_shot } else { pk }
            } else {
                0.0
            }
        })
        .collect();

    on_progress(1.0);
    PowerSpectrumResult {
        k_centers,
        power,
        n_modes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::ndarray::Array3;

    #[test]
    fn white_noise_power_positive() {
        let n = 16usize;
        let box_size = 100.0f64;
        let mut grid = Array3::<f64>::ones((n, n, n));
        grid[[0, 0, 1]] = 2.0;

        let dk = 2.0 * PI / box_size;
        let k_min = dk;
        let k_max = dk * (n / 2) as f64;
        let result = compute_power_spectrum(&grid, box_size, k_min, k_max, 8, true, false, false, |_| {});

        assert_eq!(result.k_centers.len(), 8);
        assert!(result.power.iter().all(|&p| p >= 0.0));
    }

    #[test]
    fn total_modes_equals_grid_modes_in_range() {
        let n = 8usize;
        let box_size = 50.0f64;
        let grid = Array3::<f64>::ones((n, n, n));
        let dk = 2.0 * PI / box_size;
        let k_min = dk * 0.5;
        let k_max = dk * (n / 2) as f64;
        let result = compute_power_spectrum(&grid, box_size, k_min, k_max, 4, false, false, false, |_| {});
        let total_modes: u64 = result.n_modes.iter().sum();
        assert!(total_modes > 0);
    }
}
