# Controller

## Theory and implementation

The controller converts the slopes of the WFS light paths into commands of the DMs, in the modal basis of the IM (see [Interaction Matrix Helper](interaction_matrix.md)). At every iteration `computeControlAction` returns the **absolute** command of each DM, $u_i = B_i\,c_i$, where $B_i$ are the columns of the modal basis of DM $i$ that match the IM (the first `discarded_modes` columns are skipped) and $c_i$ the modal coefficients [m].

### Conventions

- Slopes $s$ are those of the WFS after the loop delay (`LightPath.get_wavefront_error`), in the units of the IM (px, or rad). The linear model is $s = s_{turb} + H\,c$, with $H$ the IM.
- The controller works with the **negative** slopes, $e_{res} = -s$ (`error_res` in the code, "$-1$ for the feedback"), so that a positive gain drives the slopes to zero: $c \leftarrow c + g\,R\,e_{res}$.
- $d_i$ is the delay of DM $i$ in samples (`delay`, int or list, default 1). `command_history` keeps the last commands: $c_i(k-d_i)$ is the command computed $d_i$ calls ago.
- Per-DM parameters (`rcond`, `beta`, `gain`, `decay`, `ki`, `nModes`) accept a scalar (copied to all DMs) or a list of length $n_{DM}$.

### Reconstructors

Let $H$ be the IM (stacked over the WFS that are used), of shape $n_s\times n$, and $H = U\Sigma V^T$ its SVD with singular values $\sigma_1\ge\sigma_2\ge\dots$.

**Inversion.** $R = H^{+}$, the pseudo-inverse with `rcond` $\rho$ (`torch.linalg.pinv`): singular values $\sigma_j<\rho\,\sigma_1$ are set to zero, i.e. $R=V\,\mathrm{diag}(\sigma_j^{-1}\,[\sigma_j\ge\rho\sigma_1])\,U^T$. It is a fraction (not a percentage) of the largest singular value. The discarded combinations of modes are not controlled (zero gain), which limits the noise propagation of poorly sensed modes.

**Tikhonov.** $R=(H^TH+\alpha I)^{-1}H^T = V\,\mathrm{diag}\!\left(\frac{\sigma_j}{\sigma_j^2+\alpha}\right)U^T$ with $\alpha=\beta\,\sigma_1^2$ (`beta`, dimensionless, default $10^{-4}$): the regularisation is relative to the largest singular value, so $\beta$ does not depend on the units of the slopes.

**Joint block reconstructor** (`joint_reconstruction=True`, default, for inversion and tikhonov). One matrix is built over all the active WFS (rows) and all the DMs (columns)

$$
H_{glob}=\begin{pmatrix} H_{1,1} & \cdots & H_{1,n_{DM}}\\ \vdots & & \vdots\\ H_{n_{WFS},1} & \cdots & H_{n_{WFS},n_{DM}}\end{pmatrix}
$$

where $H_{w,i}$ is the IM of DM $i$ on WFS $w$, or zeros when the link is not in the target mask, and its pseudo-inverse $R_{glob}$ is split in row blocks $R_i$ (one per DM). The DMs then share the work: the slopes of a WFS that sees several DMs are explained by all of them at once. For inversion, a single $\rho$ is used for the whole matrix (the minimum of the list). For tikhonov, $\Gamma=\mathrm{diag}(\alpha_i I_{n_i})$ with $\alpha_i=\beta_i\,\sigma_{1}(H_{:,i})^2$, the largest singular value of the columns of DM $i$, and $R_{glob}=(H_{glob}^TH_{glob}+\Gamma)^{-1}H_{glob}^T$. The input of each $R_i$ is the concatenation of $-s$ of all the active WFS in light-path order, whatever the masks.

**Per-DM reconstructors** (`joint_reconstruction=False`, and always for the tomography method). Each DM is inverted independently with the IMs of the WFS of its row of the target mask. The input of $R_i$ is the concatenation of $-s$ of the WFS of the row of the control mask. A DM without WFS in its row gets a zero reconstructor (warning).

**Masks.** `control_mask` ($n_{DM}\times n_{LP}$) selects the (DM, WFS) pairs whose slopes the controller uses; `target_mask` (default `control_mask`) selects the pairs whose IM enters the reconstructor. Pairs without a measured IM are ignored (warning).

**Modes.** The number of controlled modes of DM $i$ is $\min(n_{basis}-n_{discarded},\ nModes_i,\ n_{IM\,columns})$. `discarded_modes` (2 for altitude DMs with zernike/dh/kl) are the leading modes removed at the measurement of the IM; the command uses the basis columns shifted by that offset.

### Control laws

With $e(k)=R\,e_{res}(k)$ the modal error (for the tomography method see below), per DM:

| Law | Recursion | Transfer function $C(z)$ ($c = C\,e$) |
|-----|-----------|----------------------------------------|
| `leaky` | $c(k)=g\,e(k)+a\,c(k-1)$, $a=$`decay` | $\dfrac{g}{1-a z^{-1}}$ |
| `forwardPI` | $c(k)=c(k-1)+g\,[e(k)-e(k-1)]+k_i\,e(k-1)$ | $g+\dfrac{k_i z^{-1}}{1-z^{-1}}$ |
| `backwardPI` | $c(k)=c(k-1)+g\,[e(k)-e(k-1)]+k_i\,e(k)$ | $g+\dfrac{k_i}{1-z^{-1}}$ |
| `stateSpace` | $y(k)=C x(k)+D\,u(k),\quad x(k+1)=A x(k)+B\,u(k)$, $u=e$, $c=y$ | $C(zI-A)^{-1}B+D$ |

`decay` is the leak factor applied to the previous command ($a=1$ is a pure integrator, $a<1$ leaks). For the PI laws the proportional action is applied on the variation of the error, as a velocity-form PI. The state-space matrices are given per DM, $A$: $n_x\times n_x$, $B$: $n_x\times n_{modes}$, $C$: $n_{modes}\times n_x$, $D$ optional (zeros); the state starts at zero. The loop delay is not part of $C(z)$: it comes from the light path buffer (the slopes are $d$ samples old).

### Operation modes

- `closed`: $e_{res}=-s$ with the residual slopes seen by the WFS (the DM correction is included in $s$).
- `open`: the command is zero (flat DM) at all iterations; the slopes are not used. It is meant to record open-loop telemetry or the atmosphere-only performance.
- `polc` (pseudo-open-loop control, Ellerbroek and Vogel 2003): the reconstructor acts on the estimate of the open-loop slopes

$$
s_{pol}(k)=s(k)-\sum_j H_j\,c_j(k-d_j),
$$

with the code's sign $-s_{pol}=e_{res}+\sum_j H_j c_j(k-d_j)$, where $H_j$ is the IM of DM $j$ on the slopes of the vector and $c_j(k-d_j)$ the command that the WFS is seeing now. For inversion/tikhonov with the `leaky` or PI laws the law is

$$
c(k)=(1-g)\,c(k-1)+g\,\big(-R\,s_{pol}(k)\big),
$$

whose DC gain is 1 for any $0<g\le1$ (the integrator of the closed-loop is replaced by a first-order filter, because an integrator fed with open-loop slopes has no feedback and winds up). `decay` and `ki` are not used (a warning is logged for the PI laws). A `stateSpace` law receives $-R\,s_{pol}$ as its input $u$.

The slopes whose commands are added back depend on the reconstructor. With a joint reconstructor, the global vector gets $\sum_j H_j c_j(k-d_j)$ with $H_j$ the IM of DM $j$ on all the active WFS, which requires all the DMs to have their IM on the global vector (otherwise `ValueError`). With per-DM reconstructors, the slope vector of DM $i$ (the WFS of its row of the control mask) gets the commands of **every** DM seen by those WFS, not only DM $i$. If a DM is in the optical path of one of those WFS and its IM on that WFS was not measured, the pseudo-open-loop slopes would be incomplete and a `ValueError` is raised at the first call.

*Limitation.* POLC uses the commanded $c(k-d_j)$, not the DM shape filtered by the DM dynamic model (response, hysteresis, coupling dynamics) that the WFS actually sees; the IM is also measured with the dynamics off. Using the filtered shape is left as a future development.

### Tomography

With `reconstructionMethod='tomography'`, `R_tomo` (HDF5 dataset `R_tomo` or `Rtomo` in `R_tomo_path`, see [Learn Estimator](learn_estimator.md)) maps the measured slopes to the slopes of the target (virtual) WFS, $s_{tgt}=R_{tomo}\,s_{pol}$. It is kept on the CPU. `set_tomography_targets(D_list, joint=True)` takes the IMs $D_i$ of every DM on the target slopes, builds a reconstructor $R_i$ of them (joint: Tikhonov of $D_{all}=[D_1\dots D_n]$ with the per-DM $\alpha_i$ as above, so the DMs share the fit; per-DM otherwise) and precomputes

$$
M_1^{(i)}=R_i\,R_{tomo},\qquad M_2^{(i)}=R_i\,D_{all}.
$$

The modal error of DM $i$ is the reconstruction of the target residual, $R_i\,(R_{tomo}\,s_{pol}-D_{all}\,c(k-1))$, evaluated online as

$$
e_i(k)=M_1^{(i)}\,(-s_{pol}(k))-M_2^{(i)}\,c(k-1),
$$

with $c(k-1)$ the stacked last commands, in the code's sign. It is then filtered by the control law (leaky or PI; `stateSpace` is not applied in this branch). `release_R_tomo=True` frees $R_{tomo}$ after the products are formed, so the online cost is two small matrix-vector products. Without `set_tomography_targets` the open-loop target slopes are integrated (warning). The formula assumes pseudo-open-loop slopes: it is the `polc` operation that provides them.

### Limitations

- Linear model: $H$ measured with one push per mode and without dynamics.
- No explicit noise model in inversion/tikhonov: $\rho$ and $\beta$ are tuned by hand.
- POLC with commanded, not filtered, DM shape (see above).

### References

- Ellerbroek, B. L. and Vogel, C. R. (2003), Simulations of closed-loop wavefront reconstruction for multiconjugate adaptive optics on giant telescopes, Proc. SPIE 5169, 206.
- Ellerbroek, B. L. (2002), Efficient computation of minimum-variance wavefront reconstructors with sparse matrix techniques, J. Opt. Soc. Am. A 19, 1803.
- Gendron, E. and Lena, P. (1994), Astronomical adaptive optics. I. Modal control optimization, A&A 291, 337 (modal control with gain optimisation).

::: SAOS.Controller.Controller
    options:
      show_root_heading: false
      show_root_full_path: false
      members: true
