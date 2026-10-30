# Learn Estimator

## Theory and implementation

`LearnEstimator` implements a Learn and Apply (L&A) tomographic estimator (Vidal, Gendron and Rousset 2010): the parameters of a parametric model of the covariance of the WFS slopes (turbulence profile, $r_0$, $L_0$, noise, optionally wind) are **learned** from slope telemetry, and the model is then **applied** to build a minimum-variance reconstructor $R_{tomo}$ from the measured slopes to the slopes of a set of target (virtual) WFS.

### Covariance model of the slopes

A slope is modelled as the difference of the phase at the midpoints of two opposite edges of the sub-aperture, divided by the sub-aperture size $d$: for the x slope of sub-aperture $a$, $s_x=[\phi(A)-\phi(a)]/d$, with $A=r_a+(d/2,0)$ and $a=r_a-(d/2,0)$ (the same for y). For a stationary phase with structure function $D_\phi$, $\langle\phi(r_1)\phi(r_2)\rangle=\mathrm{const}-D_\phi(|r_1-r_2|)/2$, and the covariance of two slopes with edge points $(A,a)$ and $(B,b)$ (sizes $d$, $d'$) is

$$
C_{ss'}=\frac{1}{2\,d\,d'}\sum_{l=1}^{N_L} w_l\,\Big[-D_l(A-B)+D_l(A-b)+D_l(a-B)-D_l(a-b)\Big],
$$

where $w_l$ are the fractional $C_n^2$ of the layers ($\sum w_l=1$) and $D_l$ is the structure function of layer $l$ with the total $r_0$ (so that $r_{0,l}^{-5/3}=w_l\,r_0^{-5/3}$). The points of each sub-aperture are projected to layer $l$ at altitude $h_l$ along its source direction: $r=h_l\,\theta+r_{subap}$ ($\theta$ in rad from the source coordinates), with $h_l=h_l^{zenith}/\cos z$ for the zenith angle $z$. There is no cone effect (the sources are at infinity; LGS are not supported). The covariance is block structured: the rows and columns are $[x_{WFS_1},\dots,x_{WFS_n},y_{WFS_1},\dots,y_{WFS_n}]$, and noise is added to the diagonal.

The von Karman phase structure function is evaluated with a 10-term power series in $x=\pi r/L_0$ (`vk_structure_torch`):

$$
D_\phi(r)=-k_1\,r_0^{-5/3}L_0^{5/3}\Big[a_0\,x^{5/3}+\sum_{n=1}^{10}\big(a_n\,x^{5/3}+b_n\big)x^{2n}\Big],
$$

$a_0=\Gamma(-5/6)/2^{1/6}$, $b_0=\Gamma(5/6)/2^{1/6}$, $a_n=a_{n-1}/[n(n+5/6)]$, $b_n=b_{n-1}/[n(n-5/6)]$, $k_1=\Gamma(11/6)\,2^{1/6}\pi^{-8/3}[\tfrac{24}{5}\Gamma(6/5)]^{5/6}$. For $L_0\to\infty$ it tends to the Kolmogorov $6.88\,(r/r_0)^{5/3}$. The series is accurate (relative error below about $10^{-4}$) for $x\le3$, so `learn` bounds $L_0\ge L_{0,min}=\pi\,r_{max}/3$, with $r_{max}$ the largest separation of the problem; a warning is logged if the fit sits at that bound (then $L_0$ is not constrained by the data). $D_\phi(0)=0$.

### Units

The slopes (px) are converted to phase gradients at 500 nm [rad/m]: $s\,[\mathrm{px}]\cdot \mathrm{plate\ scale}\,[\mathrm{arcsec/px}]/206265$ is the wavefront tilt [rad of angle, i.e. OPD gradient], multiplied by $2\pi/\lambda_{500}$. The fitted $r_0$ is therefore the $r_0$ at 500 nm (as in `Atmosphere`) whatever the WFS band, and the noise variance is in (rad/m)$^2$. The conversion uses the plate scale of the first measured WFS for all of them and assumes slopes in px (not rad). $R_{tomo}$ itself is a ratio of covariances and is independent of this scale: it maps slopes in some linear unit to slopes in the same unit.

### What the model does not include

- The slope is the finite difference between the two edge midpoints, not the average of the gradient over the sub-aperture: the spatial filtering of a real WFS (diffraction, pixel size, detector sampling, and for the solar Shack-Hartmann the averaging over the extended field of the sub-field images) is not modelled.
- Partial sub-apertures are treated as full ones (all `valid_subapertures`, size $d$).
- The measured-slope gain of the sensor (for example the correlating SH, which measures a fraction of the true shift) is not modelled: it is absorbed in the fitted parameters.
- Aliasing, the temporal averaging of the exposure and the dynamics of the loop; the telemetry must be open-loop or pseudo-open-loop slopes.
- Stationary, homogeneous Gaussian turbulence with one $L_0$ and a common $r_0$; the altitudes are fixed (only the fractions $w_l$ are fitted).

### Learning (`learn`)

A random fraction `selSubaps` of the sub-apertures of every WFS and a random fraction `selSamples` of the frames of the saved telemetry (`/LightPath_<i>/slopes_1D`) are used. The sample covariance matrix of the (mean-removed) slopes $\hat C=\frac{1}{N-1}\sum_k (s_k-\bar s)(s_k-\bar s)^T$ is computed, converted to phase gradients, and the parameters are found by minimising

$$
\mathcal{L}(\theta)=\frac{1}{N_s^2}\sum_{ij}\big[C(\theta)-\hat C\big]_{ij}^2 ,\qquad \theta=(r_0,L_0,w_l,\sigma_n^2),
$$

with $C(\theta)$ including the noise $\sigma_n^2$ on the diagonal (a scalar or one value per WFS, equal for x and y).

**Phase 1** (instantaneous covariance): the parameters are unconstrained variables mapped as $r_0=\mathrm{softplus}(\cdot)+10^{-6}$, $L_0=L_{0,min}+\mathrm{softplus}(\cdot)$, $w=\mathrm{softmax}(\cdot)$ (so $w_l>0$ and $\sum w_l=1$), $\sigma_n^2=\mathrm{softplus}(\cdot)+10^{-8}$ (`optimize_noise=True`; otherwise no noise term). The minimisation uses Adam (`lr`, `max_iters`), a plateau learning-rate scheduler, early stopping (`patience`) and restores the best parameters. The loss is not weighted, so the large-variance entries dominate.

**Phase 2** (`estimate_wind=True`): with frozen flow, $\phi(r,t+\tau)=\phi(r-V\tau,t)$, the covariance between the slopes at $t$ and at $t+\tau$ is the same expression with the separation shifted by the layer displacement, $|r_1-r_2+V_l\tau|$. The wind components $V_{x,l},V_{y,l}$ [m/s] of every layer are fitted with Adam, with the Phase 1 parameters frozen, to the cross-covariances at the delays `wind_delay_frames` (loss averaged over the delays), with $\tau=\mathrm{delay}\cdot T_s$ ($T_s$ the sampling time of the simulation, so the telemetry must be saved at every iteration). The sample pairs are $(t,t+\mathrm{delay})$. If `estimate_wind=False`, the phase is skipped and the reconstructor is not predictive. The initial wind is `atm_guess['windSpeedX/Y']` or zero.

### Applying (`build_reconstructor`)

The MMSE linear estimator of the target slopes $s_t$ from the measured slopes $s$ with covariances $C_{ts}=\langle s_ts^T\rangle$ and $C_{ss}=\langle ss^T\rangle$ is (Ellerbroek 2002)

$$
R_{tomo}=C_{ts}\,\big(C_{ss}+C_{n}+\varepsilon I\big)^{-1},
$$

with $C_n=\sigma_n^2 I$ and $\varepsilon=$`regularization` ($10^{-9}$, in the units of the covariance), and the matrices built with all the sub-apertures of the measured and target WFS. The system is solved by Cholesky factorisation (falls back to a general solver) in chunks of target rows, on the CPU by default or on the GPU with fallback to CPU on memory errors. Rows follow the target slopes and columns the measured ones, permuted to the controller order $[x_1,y_1,x_2,y_2,\dots]$ (`permute_for_controller=True`), so that it is the matrix `R_tomo` read by [Controller](controller.md).

If the wind is available, the reconstructor is **predictive**: the target slopes are those at $t+\Delta t$, with $\Delta t=(\mathrm{delay}+1)\,T_s$ by default (the loop latency of the target light path: the slopes of frame $k$ reach the controller after `delay` frames and the command acts from the next frame), and $C_{ts}$ uses the separation $r_t-r_s-V_l\Delta t$. Without wind it is the non-predictive L&A reconstructor.

### Limitations

- Non-convex fit with an unweighted loss and a limited number of samples: the result depends on the initial guess and on the random selection; $L_0$ and the layer fractions are weakly constrained, and the altitudes are not fitted.
- The model errors above (solar field averaging, partial sub-apertures, sensor gain) bias the fitted parameters.
- $C_{ss}$ is dense: the cost of $R_{tomo}$ is cubic in the number of measured slopes, and the matrix can need several GB for large systems.

### References

- Vidal, F., Gendron, E. and Rousset, G. (2010), Tomography approach for multi-object adaptive optics, J. Opt. Soc. Am. A 27, A253.
- Ellerbroek, B. L. (2002), Efficient computation of minimum-variance wavefront reconstructors with sparse matrix techniques, J. Opt. Soc. Am. A 19, 1803.

::: SAOS.tomography.learnEstimator.LearnEstimator
    options:
      show_root_heading: false
      show_root_full_path: false
      members: true
