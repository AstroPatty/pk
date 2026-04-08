use numpy::ndarray::Array3;

/// Cloud-In-Cell (CIC) mass assignment.
///
/// Deposits equal-weight particles onto a uniform cubic grid using trilinear
/// (CIC) interpolation. Periodic boundary conditions are applied.
///
/// # Arguments
/// * `positions` - Particle positions in physical coordinates `[0, box_size)`.
/// * `box_size`  - Side length of the simulation box (same units as positions).
/// * `n_cells`   - Number of grid cells per side.
///
/// # Returns
/// `Array3<f64>` of shape `[n_cells, n_cells, n_cells]` containing the CIC counts.
pub fn cic_deposit(positions: &[[f64; 3]], box_size: f64, n_cells: usize) -> Array3<f64> {
    let n = n_cells;
    let mut grid = Array3::<f64>::zeros((n, n, n));
    let inv_spacing = n as f64 / box_size;

    for pos in positions {
        // Convert to grid coordinates.
        let gx = pos[0] * inv_spacing;
        let gy = pos[1] * inv_spacing;
        let gz = pos[2] * inv_spacing;

        // Lower-left cell index.
        let ix = gx.floor() as isize;
        let iy = gy.floor() as isize;
        let iz = gz.floor() as isize;

        // Fractional offset within the cell: in [0, 1).
        let dx = gx - ix as f64;
        let dy = gy - iy as f64;
        let dz = gz - iz as f64;

        // CIC weights along each axis: [lower, upper].
        let wx = [1.0 - dx, dx];
        let wy = [1.0 - dy, dy];
        let wz = [1.0 - dz, dz];

        // Deposit onto the 8 surrounding cells with periodic wrapping.
        let n_isize = n as isize;
        for (di, &wxi) in wx.iter().enumerate() {
            let cx = ((ix + di as isize).rem_euclid(n_isize)) as usize;
            for (dj, &wyj) in wy.iter().enumerate() {
                let cy = ((iy + dj as isize).rem_euclid(n_isize)) as usize;
                for (dk, &wzk) in wz.iter().enumerate() {
                    let cz = ((iz + dk as isize).rem_euclid(n_isize)) as usize;
                    grid[[cx, cy, cz]] += wxi * wyj * wzk;
                }
            }
        }
    }

    grid
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn total_counts_conserved() {
        let positions = vec![[0.5, 0.5, 0.5], [1.5, 2.5, 3.5], [3.9, 0.1, 1.9]];
        let grid = cic_deposit(&positions, 4.0, 4);
        let total: f64 = grid.iter().sum();
        assert!(
            (total - positions.len() as f64).abs() < 1e-12,
            "expected {}, got {total}",
            positions.len()
        );
    }

    #[test]
    fn node_deposits_to_one_cell() {
        // A particle exactly on a grid node (integer grid coord) goes entirely to that cell.
        // [0.0, 0.0, 0.0] -> gx=gy=gz=0, dx=dy=dz=0 -> wx=wy=wz=[1,0] -> all weight to [0,0,0].
        let positions = vec![[0.0, 0.0, 0.0]];
        let grid = cic_deposit(&positions, 4.0, 4);
        assert!((grid[[0, 0, 0]] - 1.0).abs() < 1e-12);
        assert_eq!(grid[[1, 0, 0]], 0.0);
    }

    #[test]
    fn periodic_wrapping() {
        // Particle at x=3.75, y=z=0 (on node in y,z).
        // gx=3.75 -> ix=3, dx=0.75 -> wx=[0.25, 0.75].
        // Cell 3 gets 0.25, wrapped cell 0 gets 0.75.
        let positions = vec![[3.75, 0.0, 0.0]];
        let grid = cic_deposit(&positions, 4.0, 4);
        assert!((grid[[3, 0, 0]] - 0.25).abs() < 1e-12);
        assert!((grid[[0, 0, 0]] - 0.75).abs() < 1e-12);
    }
}
