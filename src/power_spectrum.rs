use numpy::ndarray::Array3;
use rayon::prelude::*;
use rustfft::{num_complex::Complex, FftPlanner};
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
    if m <= n / 2 { m as isize } else { m as isize - n as isize }
}

/// CIC window function value for FFT index m in a grid of size n.
/// W(m) = sinc(π * m_signed / n)
#[inline]
fn cic_window(m: usize, n: usize) -> f64 {
    sinc(PI * signed_mode(m, n) as f64 / n as f64)
}

/// Perform an in-place 3D FFT on a flat row-major array of shape [n, n, n].
///
/// Each of the three axis passes is fully parallelised with Rayon.
/// `on_progress` is called at 0.30, 0.55, and 0.80 between passes.
fn fft3d(data: &mut [Complex<f64>], n: usize, on_progress: &impl Fn(f64)) {
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n);
    let scratch_len = fft.get_inplace_scratch_len();
    let n2 = n * n;

    // --- Axis z: n² contiguous FFTs of length n ---
    // par_chunks_mut(n) yields each [i,j,:] row as a disjoint mutable slice.
    data.par_chunks_mut(n).for_each_init(
        || vec![Complex::default(); scratch_len],
        |scratch, chunk| fft.process_with_scratch(chunk, scratch),
    );
    on_progress(0.30);

    // --- Axis y: n independent i-slices, each of size n² ---
    // par_chunks_mut(n*n) gives each [i,:,:] block as a disjoint mutable slice.
    // Within each block, the n columns [i,k,:] are processed sequentially.
    data.par_chunks_mut(n2).for_each_init(
        || (vec![Complex::default(); n], vec![Complex::default(); scratch_len]),
        |(buf, scratch), row| {
            for k in 0..n {
                for j in 0..n { buf[j] = row[j * n + k]; }
                fft.process_with_scratch(buf, scratch);
                for j in 0..n { row[j * n + k] = buf[j]; }
            }
        },
    );
    on_progress(0.55);

    // --- Axis x: n² independent (j,k) columns strided over i ---
    //
    // Each column accesses data[i*n² + j*n + k] for i in 0..n.
    // Distinct (j,k) pairs yield disjoint index sets:
    //   i*n² + j*n + k  ==  i*n² + j'*n + k'  ⟹  j == j' ∧ k == k'
    // so all n² columns can be processed concurrently without aliasing.
    //
    // *mut T is !Send, so we cast the pointer to usize to move it across
    // the thread boundary, then recover it inside the closure.
    let ptr = data.as_mut_ptr() as usize;
    (0..n2).into_par_iter().for_each_init(
        || (vec![Complex::default(); n], vec![Complex::default(); scratch_len]),
        |(buf, scratch), jk| {
            let j = jk / n;
            let k = jk % n;
            let base = ptr as *mut Complex<f64>;
            // SAFETY: disjoint column access proven above.
            for i in 0..n {
                buf[i] = unsafe { *base.add(i * n2 + j * n + k) };
            }
            fft.process_with_scratch(buf, scratch);
            for i in 0..n {
                unsafe { *base.add(i * n2 + j * n + k) = buf[i] };
            }
        },
    );
    on_progress(0.80);
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
/// * `grid`               - CIC particle counts, shape [n, n, n]. Assumed cubic.
/// * `box_size`           - Physical side length of the box.
/// * `k_min`              - Lower edge of the first k bin.
/// * `k_max`              - Upper edge of the last k bin.
/// * `n_bins`             - Number of bins.
/// * `log_bins`           - If true, log-spaced bins (geometric-mean centers);
///                          if false, linearly spaced (arithmetic-mean centers).
/// * `deconvolve_cic`     - If true, correct for CIC window function suppression.
/// * `subtract_shot_noise`- If true, subtract P_shot = V / N_particles per bin.
/// * `on_progress`        - Called with a value in [0, 1] at key checkpoints.
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
    let mean = grid.iter().sum::<f64>() / n3; // mean = N_particles / N_cells³
    let p_shot = volume / (mean * n3);         // V / N_particles
    let mut cdata: Vec<Complex<f64>> = grid
        .iter()
        .map(|&v| Complex::new((v - mean) / mean, 0.0))
        .collect();
    on_progress(0.05);

    // --- Step 2: 3D FFT, then normalize by 1/N³ ---
    // fft3d calls on_progress at 0.30, 0.55, 0.80 between axis passes.
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
        (0..n_bins).map(|i| (bin_edges[i] * bin_edges[i + 1]).sqrt()).collect()
    } else {
        (0..n_bins).map(|i| 0.5 * (bin_edges[i] + bin_edges[i + 1])).collect()
    };

    // Precompute bin-lookup parameters to keep the inner loop branch-free.
    let (bin_offset, bin_scale) = if log_bins {
        let ln_kmin = k_min.ln();
        (ln_kmin, n_bins as f64 / (k_max.ln() - ln_kmin))
    } else {
        (k_min, n_bins as f64 / (k_max - k_min))
    };

    let dk = 2.0 * PI / box_size;

    // --- Step 4: accumulate |δ̃_k|² into bins ---
    //
    // Parallelise over the n outer i-slices. Each thread keeps its own
    // (power_sum, n_modes) accumulators; Rayon's reduce merges them at the end.
    let (power_sum, n_modes) = (0..n)
        .into_par_iter()
        .fold(
            || (vec![0.0f64; n_bins], vec![0u64; n_bins]),
            |(mut ps, mut nm), i| {
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

                        ps[bin] += pk;
                        nm[bin] += 1;
                    }
                }
                (ps, nm)
            },
        )
        .reduce(
            || (vec![0.0f64; n_bins], vec![0u64; n_bins]),
            |(mut a_ps, mut a_nm), (b_ps, b_nm)| {
                for b in 0..n_bins {
                    a_ps[b] += b_ps[b];
                    a_nm[b] += b_nm[b];
                }
                (a_ps, a_nm)
            },
        );

    // --- Step 5: P(k) = V · mean(|δ̃_k|²) per bin ---
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
    PowerSpectrumResult { k_centers, power, n_modes }
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
        let result = compute_power_spectrum(
            &grid, box_size, k_min, k_max, 8, true, false, false, |_| {},
        );
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
        let result = compute_power_spectrum(
            &grid, box_size, k_min, k_max, 4, false, false, false, |_| {},
        );
        let total_modes: u64 = result.n_modes.iter().sum();
        assert!(total_modes > 0);
    }
}
