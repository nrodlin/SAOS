# `test_modal_basis.py` — modal bases for DM commands and interaction matrices (T1)

Source: `tests/test_modal_basis.py`

Covers `SAOS.modalBasis.zonalModes.generate_zonal_modes`, `discHarmonicModes.generate_dh_modes`, `karhunenLoeveModes.generate_kl_modes`, `hadamardModes.generate_hadamard_modes` and `zernikeModes.generate_zernike_modes` / `ZernikeNaive.R_nm_Jacobi`. Bases are evaluated on the actuator coordinates of an 11x11 cartesian ground `DeformableMirror` (97 valid actuators, 32 px pupil, only the actuator geometry is used) or of an equivalent cartesian grid (e.g. 37x37 for the EST ASM) where only coordinates are needed; columns are modes normalised to $\max|\cdot| = 1$. The strategy is structural (rank, orthogonality, piston content) plus comparison with independent definitions written in the test (Noll ordering and Zernike formula through Jacobi polynomials, Noll Kolmogorov covariance, Neumann eigenfunctions of the Laplacian on the disk), mostly through subspace distances. The bases are deterministic, so tolerances (`tests/tolerances/modal_basis.yaml`) are numerical, not statistical. Whole module `fast`.

Independent references defined in the file: Noll ordering $j\to(n,m,\cos|\sin)$; Zernike polynomials

$$Z_j = N_{nm}\,R_n^{|m|}(\rho)\,\{\cos,\sin\}(m\theta),\quad N_{nm}=\sqrt{n+1}\ (m=0),\ \sqrt{2(n+1)}\ (m\neq0),\quad R_n^m(\rho) = (-1)^k\rho^m P_k^{(m,0)}(1-2\rho^2),\ k=\tfrac{n-m}{2};$$

and the Neumann eigenfunctions $J_m(k\rho)\{\cos,\sin\}(m\theta)$ with $J_m'(k)=0$ used for disk harmonics.

## `test_basis_spans_actuator_space`
- **Module:** `generate_zonal_modes`, `generate_dh_modes`, `generate_kl_modes`, `generate_hadamard_modes`.
- **Tier:** fast. Parametrised over `name` = `zonal`, `dh`, `kl`, `hadamard`.
- **What it checks:** the basis restricted to the valid actuators has rank equal to its number of modes (singular values above $10^{-10}s_{max}$). For the piston-free bases (dh, kl, hadamard) each mode has $|\mathrm{mean}|/\mathrm{rms}$ below tolerance and the number of modes is at most $n_{valid}-1$. The zonal basis is the identity on the valid actuators.
- **Tolerance:** `modal_basis.basis`: `rank_rel_sv` $10^{-10}$, `piston_fraction` $10^{-10}$ (piston-free modes measure about $10^{-16}$).
- **References:** none.

## `test_orthogonal_bases_have_diagonal_gram`
- **Module:** `generate_zonal_modes`, `generate_dh_modes`, `generate_kl_modes` (QR-orthonormalised).
- **Tier:** fast. Parametrised over `name` = `zonal`, `dh`, `kl`.
- **What it checks:** the Gram matrix $B^TB$ on the valid actuators is diagonal: $\max|G_{ij}|_{i\ne j}/\max G_{ii}$.
- **Tolerance:** `modal_basis.basis.gram_offdiag` = $10^{-10}$ (measured about $3\times10^{-16}$ for DH/KL).
- **References:** none.

## `test_zernike_rank_limited_by_actuator_sampling`
- **Module:** `generate_zernike_modes` (default `nModes` = number of valid actuators).
- **Tier:** fast.
- **What it checks:** for the 11x11 DM and a 37x37 grid (EST ASM size, threshold 0.7533 pitch) the first half of the modes is full rank and the piston fraction is below tolerance; the rank with `nModes` = `nValidAct` is only recorded. As the docstring states, high-order Zernikes alias on the actuator grid and the full basis is rank deficient, a property of the sampling and not of the polynomial evaluation, so it is recorded and not failed (IMs with the full Zernike basis contain dependent columns that only regularisation removes).
- **Tolerance:** rank deficiency of the first half exactly 0 (singular values above $10^{-10}s_{max}$, `basis.rank_rel_sv`); `basis.piston_fraction` $10^{-10}$.
- **References:** Noll 1976 (Zernike modes); sampling argument is generic.

## `test_noll_covariance_reference_matches_noll_table`
- **Module:** none from SAOS; sanity check of the test's own Noll covariance reference.
- **Tier:** fast.
- **What it checks:** the variances of $Z_2\ldots Z_{11}$ from the implemented covariance, relative to $Z_2$, equal $(\Delta_{j-1}-\Delta_j)/(\Delta_1-\Delta_2)$ from Noll's table of residual variances. The overall constant of the formula is 1.3 % above the table (0.4536 against 0.4479 for $Z_2$), irrelevant for eigenvectors; recorded.
- **Theory / assumptions:** Kolmogorov covariance of Noll-normalised coefficients (Noll 1976 eq. 16), in units of $(D/r_0)^{5/3}$:

$$\langle a_j a_{j'}\rangle = K\,(-1)^{(n+n'-2m)/2}\sqrt{(n+1)(n'+1)}\;\frac{\Gamma\!\left(\frac{n+n'-5/3}{2}\right)}{\Gamma\!\left(\frac{n-n'+17/3}{2}\right)\Gamma\!\left(\frac{n'-n+17/3}{2}\right)\Gamma\!\left(\frac{n+n'+23/3}{2}\right)}$$

  for $m=m'$ and equal parity of $j$ (or $m=0$), zero otherwise; $K=\Gamma(14/3)\left[\tfrac{24}{5}\Gamma(\tfrac65)\right]^{5/6}\Gamma(\tfrac{11}{6})^2/(2\pi^2)$.
- **Tolerance:** absolute $10^{-3}$ on the variance ratios (covers the 4-digit rounding of the table; hard-coded).
- **References:** Noll 1976, JOSA 66:207 (eq. 16 and table IV).

## `test_kl_modes_are_the_atmospheric_eigenmodes_of_the_zernike_subspace`
- **Module:** `generate_kl_modes` (first 20 modes of an 11x11 DM).
- **Tier:** fast.
- **What it checks:** (1) the KL modes lie in the span of piston-free Zernikes sampled on the valid actuators (Noll 2 to 21, radial orders 1 to 5 complete); (2) the variance of each mode, $\mathbf a^TC\mathbf a$, equals the eigenvalue of the same rank, and variances are non-increasing; (3) each cluster of degenerate eigenvalues (relative gap $10^{-6}$) spans the same subspace as the independent eigenvectors; (4) the modes are orthogonal on the actuators and their amplitudes are uncorrelated (covariance diagonal).
- **Theory / assumptions:** double diagonalisation. With $Z=QR$ (QR of the independent Noll-normalised Zernikes on the actuators) and $C$ the Noll covariance of the coefficients, the KL modes are $QU$ where $U$ diagonalises $R\,C\,R^T$, ordered by decreasing variance. Kolmogorov statistics; finite sampled pupil.
- **Tolerance:** `modal_basis.kl`: `span` $10^{-8}$, `eig_rel` $10^{-8}$ (variance ratio and covariance diagonality), `subspace` $10^{-6}$ (spectral norm of the projector difference per cluster; also the Gram orthogonality bound).
- **References:** Noll 1976 (covariance); none beyond Noll 1976; the KL construction is the test's own.

## `test_dh_lowest_modes_span_the_lowest_neumann_eigenfunctions`
- **Module:** `generate_dh_modes`.
- **Tier:** fast. Parametrised over `count` = 2, 5.
- **What it checks:** the first `count` DH modes (2: the $k=1.841$ pair; 5: plus the 3.054 pair and the $k=3.832$ axisymmetric mode; complete eigenvalue levels) span the `count` lowest eigenfunctions of the independent construction evaluated on the valid actuators of an 11x11 DM (disk radius is the outermost valid actuator, piston removed on the actuators), measured as the spectral norm of the difference of the projectors.
- **Theory / assumptions:** disk harmonics are the eigenfunctions of the Laplacian on the disk with zero normal derivative at the edge, $J_m(k\rho)\{\cos,\sin\}(m\theta)$ with $J_m'(k)=0$, sorted by eigenvalue $k$; the constant ($m=0$, $k=0$) is excluded.
- **Tolerance:** `modal_basis.dh.subspace` = $10^{-8}$.
- **References:** Milton & Lloyd-Hart 2005 (disk harmonics for adaptive optics); standard Neumann problem on the disk.

## `test_dh_radial_order_shells_span_documented_functions`
- **Module:** `generate_dh_modes`.
- **Tier:** fast. Parametrised over `nmax` = 1, 2, 3, 4.
- **What it checks:** the documented Noll-like labelling of SAOS disk harmonics: the first modes completing radial orders $1..n_{max}$ span exactly the functions $J_m(kr)\{\cos,\sin\}(m\theta)$ with $k$ the $q$-th root of $J_m'$, $q=(n-|m|)/2+1$ for $m\ne0$ and $n/2$ for $m=0$. The distance to the lowest-eigenvalue set (the labelling is not eigenvalue-ordered) is informational.
- **Tolerance:** `modal_basis.dh.subspace` = $10^{-8}$.
- **References:** Milton & Lloyd-Hart 2005.

## `test_zernike_low_order_shapes`
- **Module:** `generate_zernike_modes`.
- **Tier:** fast. Parametrised over Noll index 2 (tip, $\rho\cos\theta\to x$), 4 (defocus, $2\rho^2-1$), 6 (astigmatism, $x^2-y^2$), 11 (spherical, $6\rho^4-6\rho^2+1$).
- **What it checks:** the SAOS mode with the given Noll index, on the valid actuators with piston removed, is positively correlated with the analytic shape (correlation above 0.999). Fixes the sign convention for even-$j$ (cosine) and $m=0$ modes.
- **Tolerance:** `modal_basis.zernike.shape_correlation` = 0.999 (measured $1-10^{-15}$ for exact polynomials).
- **References:** Noll 1976.

## `test_zernike_noll_sign_convention`
- **Module:** `generate_zernike_modes`.
- **Tier:** fast. Parametrised over Noll index 3 ($2\rho\sin\theta\to y$) and 5 ($\sqrt6\rho^2\sin2\theta\to xy$).
- **What it checks:** the odd-$j$ (sine) modes follow Noll's sign convention: positive correlation with $+y$ and $+xy$ (correlation above 0.999), with $y$ = +row of the actuator coordinates.
- **Tolerance:** `modal_basis.zernike.shape_correlation` = 0.999.
- **References:** Noll 1976.

## `test_zernike_radial_polynomial_low_orders`
- **Module:** `ZernikeNaive.R_nm_Jacobi`.
- **Tier:** fast.
- **What it checks:** the SAOS radial polynomial (Jacobi form, the evaluator used to build the basis) matches the scipy-Jacobi reference for every $n\le30$ and all allowed $m$ on 501 points of $\rho\in[0,1]$.
- **Tolerance:** `modal_basis.zernike.radial_abs_low_orders` = $10^{-5}$ (DMs up to about 22x22 actuators).
- **References:** standard Jacobi representation of $R_n^m$ (e.g. Born & Wolf).

## `test_zernike_radial_jacobi_evaluator`
- **Module:** `ZernikeNaive.R_nm_Jacobi`.
- **Tier:** fast.
- **What it checks:** the Jacobi evaluator is accurate up to $n=49$ (sampled orders $n=0,7,\ldots,49$). The physical bound $|R|\le1$ makes the absolute criterion meaningful.
- **Tolerance:** `modal_basis.zernike.radial_abs` = $10^{-6}$ (the evaluator gives about $10^{-13}$ at $n=49$).
- **References:** none.

## `test_zernike_radial_polynomial_est_orders`
- **Module:** `generate_zernike_modes` for an EST-size DM (37x37, Zernike up to $n\approx49$).
- **Tier:** fast.
- **What it checks:** the basis generated by SAOS equals, column by column, the same basis rebuilt with `R_nm_Jacobi` temporarily replaced by the module's independent radial reference (patched in a `try/finally`); the number of columns differing by more than the bound must be 0. The numerical rank ratio (singular values above $10^{-8}s_{max}$) is recorded as information (aliasing on the grid at the highest orders is a physical limit). A plot of the radial error against $n=31..49$ is produced.
- **Tolerance:** `modal_basis.zernike.basis_column_abs` = 0.01 on the maximum column difference (modes are normalised to $\max|Z|=1$); `radial_abs` $10^{-6}$ is the reference line of the plot.
- **References:** none.
