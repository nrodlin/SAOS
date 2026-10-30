# Interaction Matrix Helper

## Theory and implementation

`InteractionMatrixHandler` measures, for every DM and every light path with a WFS that sees it, the matrix that maps a modal command of the DM to the WFS slopes, and manages the modal bases and the files that store them.

### Linear model

Let $c\in\mathbb{R}^{n}$ be the modal coefficients of a DM, $B\in\mathbb{R}^{n_{act}\times n}$ its modal basis (one mode per column, defined on the actuators; the actuator commands are $u = Bc$, in metres of surface/OPD as defined by the DM model) and $s\in\mathbb{R}^{n_s}$ the slopes of a WFS (first half x, second half y; in pixels, or in rad if the WFS has `unit_in_rad`). The controller assumes the linear model

$$
s = H\,c, \qquad H_{:,j} = \frac{s(\,\tau\, b_j\,)}{\tau},
$$

where $b_j$ is mode $j$ and $\tau$ the `stroke` [m]. The IM $H$ has one column per mode. The units of $H$ are slope units per metre of command.

### Measurement procedure (`measure`)

1. The DMs seen by each WFS light path are listed once (`initialize_im_class`): a DM object shared by several light paths is scanned once, and two distinct DM objects are always scanned separately, even with identical properties. A boolean matrix $nLP\times nDM$ marks which (light path, DM) pairs need an IM.
2. For each DM, for each mode $j$: the command $\tau\,b_j$ is applied with `updateDMShape(..., dynamicResponse=False)`, i.e. with the DM dynamic model off (the commanded shape is applied directly), all the light paths are propagated with `interaction_matrix=True` and the slopes of every affected WFS are divided by $\tau$. The DM is set to zero before the next DM.
3. During this propagation the atmosphere, vibrations and local seeing are replaced by zeros (see [Light Path](light_path.md)). No zero-command reference is subtracted: the method assumes that the slopes of the unperturbed system are zero (flat WFS reference).

The measurement is **push-only** (one-sided), by design: the simulator has no noise, no nonlinearity of the actuators and the WFS are used in their linear regime, so $s(-\tau b_j)=-s(\tau b_j)$ and a push-pull average would give the same matrix at twice the cost. The stroke only has to stay in the linear range of the WFS: the maximum displacement $\max|H|\,\tau$ is reported in the log and stored (`maxDisplacement`) as a check. A stroke large enough to leave the linear range of the WFS biases $H$. `stroke` must be a Python `float` (or a list of floats, one per DM); an `int` raises `TypeError`.

For DMs at altitude ($>0$ m) with the `zernike`, `dh` and `kl` bases, the first two modes (tip and tilt) are discarded (a warning is logged); `discarded_modes` $=2$ is stored with the IM and used by the controller to select the matching columns of the basis (see [Controller](controller.md)). `nModes` truncates the number of modes per DM before this discard.

### Modal bases (`generate_modal_basis`)

For every DM five bases are generated, with entries scaled so that each mode is bounded by $[-1,1]$ on the actuators:

| Basis | Modes | Remarks |
|-------|-------|---------|
| `zonal` | $n_{valid}$ | Identity on the valid actuators (includes piston). |
| `zernike` | $n_{valid}-1$ | Noll-ordered Zernike polynomials sampled at the actuators, piston-free, each scaled to unit rms over the valid actuators and then to unit peak. |
| `kl` | $n_{valid}-1$ | Karhunen-Loeve: Zernike basis orthonormalised on the actuators, then diagonalised with the Kolmogorov Zernike covariance (double diagonalisation), so that the modes are statistically independent. Not rescaled by $r_0$ or $D$. |
| `dh` | $n_{valid}-1$ | Disc harmonics (Milton and Lloyd-Hart 2005), orthonormalised and piston-free. |
| `hadamard` | $n_{valid}-1$ | Rows of a Sylvester Hadamard matrix of size the next power of two, piston removed. |

A warning is logged if a basis is numerically rank deficient (`check_basis_rank`, singular values below $10^{-8}$ of the largest count as zero), since linearly dependent modes cannot be controlled independently.

### BASIS_CONVENTION

The numerical content of the bases changed in the past (Noll sign of the sine Zernikes, and piston-free bases of $n_{valid}-1$ modes). `BASIS_CONVENTION` (currently 2) labels the version; it is written to the files and checked on load. An IM is only valid with the basis it was measured with: with a different convention the sine Zernike modes have the opposite sign and the feedback of those modes would be inverted. Mismatches raise `ValueError` (see below).

### File layout

IM file (`save_IM`): root attributes `nLPS`, `nDMs`, `basis_convention`; groups `LP<i>` (attributes `wavelength`, `zenith`, `azimuth`, `nSignal`) containing `IM<j>` groups (attributes `nValidAct`, `nAct`, `altitude`, `mechCoupling`, `modalBasis`, `maxDisplacement`, `discarded_modes`; dataset `data`, shape $n_s\times n_{modes}$). Modal basis file (`save_modalBasis`): root attribute `basis_convention`; a group per basis (`zonal`, `zernike`, ...) with a subgroup `DM<j>` (dataset `data`, attributes `nAct`, `nValidAct`). The slope units of the IM (`slopes_units`) are kept in memory but are not written to the file, and `load_IM` sets them to `px`.

### Saving: no silent overwrite

`save_IM(filename, overwrite=False)` and `save_modalBasis(filename, overwrite=False)` never modify an existing file by default:

- If `filename` (with `.h5` added when missing) exists and `overwrite=False`, they raise `FileExistsError` and the existing file is left untouched. Previously the file was opened in append mode.
- With `overwrite=True` the content is written to a temporary file in the same folder (`<name>.tmp<pid>.h5`) which is renamed over the target with `os.replace`, an atomic replacement on POSIX: the old file is only replaced when the new one is complete, and a failure leaves no partial file (the temporary file is removed).
- `check_output_file(filename, overwrite=False)` performs the same check without writing: call it before `measure()` so that an existing file is detected before spending the measurement time. It returns the path with the `.h5` extension.

```python
imh.check_output_file('~/simulations/im/est_im.h5')     # raises FileExistsError early
imh.measure('kl', stroke=1e-8)
imh.save_IM('~/simulations/im/est_im.h5')
imh.save_modalBasis('~/simulations/im/est_basis.h5')
```

### Loading

`load_IM` checks the convention and matches each `LP<i>` group to the light path of the same wavelength, zenith, azimuth and number of slopes (the group of the same index first), and each `IM<j>` to a DM of the same `nValidAct`, `altitude` and `nAct` (index first, never two IMs on one DM). `load_modalBasis` requires the five bases and the same `nAct`, `nValidAct` per DM. `ValueError` is raised if the IM and the modal basis have different conventions; a single file of an old convention only produces a warning.

### Limitations

- One-sided linear IM: valid in the linear regime only, no zero-command reference.
- Measured with the DM dynamic model off and without atmosphere: the loop dynamics and the DM response are not part of $H$.

### References

- Milton, N. M. and Lloyd-Hart, M. (2005), Disk harmonic functions for adaptive optics simulations, Proc. Adaptive Optics: Analysis and Methods (OSA).
- Roddier, N. (1990), Atmospheric wavefront simulation using Zernike polynomials, Opt. Eng. 29, 1174.
- Noll, R. J. (1976), Zernike polynomials and atmospheric turbulence, J. Opt. Soc. Am. 66, 207.

## IM and modal basis files

`save_IM` and `save_modalBasis` write the attribute `basis_convention` (`BASIS_CONVENTION`, currently 2: Noll sign of the
sine Zernikes and piston-free bases of `nValidAct - 1` modes, since 2026-10-27). Files without it are convention 1. An IM
is only valid with a modal basis of its own convention: `load_IM` and `load_modalBasis` raise `ValueError` when they do not
match (the sine Zernikes changed sign, so mixing them would invert the feedback of those modes). Load the modal basis file
saved with the IM, or measure the IM again.

`load_IM` puts every IM back in its own (DM, light path) entry: the groups `LP<i>/IM<j>` of the file are matched by index
first, so identical light paths or DMs of identical properties are not mixed up.

::: SAOS.InteractionMatrixHandler.InteractionMatrixHandler
    options:
      show_root_heading: false
      show_root_full_path: false
      members: true