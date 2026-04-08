mod cic;
mod power_spectrum;

use numpy::ndarray::Array3;
use numpy::{IntoPyArray, PyArray1, PyArray3, PyReadonlyArray3};
use pyo3::prelude::*;

/// Cloud-In-Cell mass assignment onto a cubic grid.
///
/// Parameters
/// ----------
/// positions : list of [float, float, float]
///     Particle positions in physical coordinates [0, box_size).
/// box_size : float
///     Side length of the simulation box.
/// n_cells : int
///     Number of grid cells per side.
///
/// Returns
/// -------
/// numpy.ndarray, shape (n, n, n), dtype float64
///     CIC counts grid.
#[pyfunction]
fn cic_deposit<'py>(
    py: Python<'py>,
    positions: Vec<[f64; 3]>,
    box_size: f64,
    n_cells: usize,
) -> Bound<'py, PyArray3<f64>> {
    let grid: Array3<f64> = cic::cic_deposit(&positions, box_size, n_cells);
    grid.into_pyarray(py)
}

/// Compute the 3D matter power spectrum P(k) from a CIC density grid.
///
/// Parameters
/// ----------
/// grid : numpy.ndarray, shape (n, n, n), dtype float64
///     CIC particle counts, as returned by ``cic_deposit``.
/// box_size : float
///     Physical side length of the box.
/// k_min : float
///     Lower edge of the first k bin.
/// k_max : float
///     Upper edge of the last k bin.
/// n_bins : int
///     Number of bins.
/// log_bins : bool
///     If True, bins are log-spaced and k_centers are geometric means.
///     If False, bins are linearly spaced and k_centers are arithmetic means.
/// deconvolve_cic : bool
///     If True, correct each Fourier mode for the CIC window function
///     suppression before computing the power spectrum.
/// subtract_shot_noise : bool
///     If True, subtract the Poisson shot noise floor P_shot = V / N_particles
///     from every bin. Removes the white-noise floor caused by discrete sampling.
/// progress_fn : callable(float) or None
///     Called with a value in [0, 1] at key checkpoints. Use with tqdm:
///
///         pbar = tqdm(total=100)
///         pk.compute_power_spectrum(..., progress_fn=lambda p: pbar.update(int(p*100) - pbar.n))
///
/// Returns
/// -------
/// k_centers : numpy.ndarray, shape (n_bins,)
///     Center k value of each bin.
/// power : numpy.ndarray, shape (n_bins,)
///     P(k) in units of box_size^3.
/// n_modes : numpy.ndarray, shape (n_bins,), dtype uint64
///     Number of Fourier modes averaged in each bin.
#[pyfunction]
#[pyo3(signature = (grid, box_size, k_min, k_max, n_bins, log_bins, deconvolve_cic, subtract_shot_noise=false, progress_fn=None))]
fn compute_power_spectrum<'py>(
    py: Python<'py>,
    grid: PyReadonlyArray3<'py, f64>,
    box_size: f64,
    k_min: f64,
    k_max: f64,
    n_bins: usize,
    log_bins: bool,
    deconvolve_cic: bool,
    subtract_shot_noise: bool,
    progress_fn: Option<Py<PyAny>>,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<u64>>,
)> {
    let grid_view = grid.as_array();
    let result = power_spectrum::compute_power_spectrum(
        &grid_view.to_owned(),
        box_size,
        k_min,
        k_max,
        n_bins,
        log_bins,
        deconvolve_cic,
        subtract_shot_noise,
        |p| {
            if let Some(ref f) = progress_fn {
                let _ = f.call1(py, (p,));
            }
        },
    );
    Ok((
        result.k_centers.into_pyarray(py),
        result.power.into_pyarray(py),
        result.n_modes.into_pyarray(py),
    ))
}

#[pymodule]
fn pk(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(cic_deposit, m)?)?;
    m.add_function(wrap_pyfunction!(compute_power_spectrum, m)?)?;
    Ok(())
}
