# Telescope

## Theory and implementation

The telescope is a pixelated pupil mask on an $N\times N$ grid ($N$ = `resolution`) with pixel size $\delta = D/N$ (`pixelSize`). It also holds the quantities shared by the other modules: the AO sampling time $T_s$ (which drives the atmosphere motion, the vibrations and the local-seeing sequences), the field of view (FoV, arcsec; `fov_rad` = FoV/206265, which sets the size of the atmospheric metapupil) and the reflectivity map.

### Pupil mask

When no pupil is given, the pixel coordinates are $x_k = -N/2 + k\,N/(N-1)$, $k=0,\dots,N-1$ (in units of one nominal pixel; the actual step is $N/(N-1)$ nominal pixels, so the mask spans $\approx N$ pixels), and with $r^2 = x^2+y^2$ and $R = (N+1)/2$

$$
P(x,y) = \begin{cases} 1 & \epsilon^2 R^2 \le r^2 < R^2 \\ 0 & \text{otherwise}\end{cases}
$$

where $\epsilon$ is `centralObstruction`, the obstruction **diameter** as a fraction of $D$. The mask is binary (no anti-aliasing). A user pupil replaces it as is and then the central obstruction is ignored. The reflectivity is `pupil * pupilReflectivity` (a scalar or a map, treated as an amplitude map; the OPD does not depend on it). The effective collecting area used for the photon budget is

$$
A_{eff} = \frac{\sum P}{N^2}\,D^2 \quad [\mathrm{m^2}],
$$

so obstruction and spiders reduce the photon flux (`photons = flux * A_eff * T_int * ...`, see [Source](source.md)).

### Spiders

`apply_spiders` resets the pupil and removes, for each arm $i$, a half-strip of width $w$ starting at the centre. With pixel centres $x_j = (j-(N-1)/2)\,D/N$, $x$ = column and $y$ = row (the frame of the source azimuths), and the arm angle $\alpha_i$ measured counter-clockwise from $+x$ in that frame,

$$
u = (x+o_x)\cos\alpha_i + (y+o_y)\sin\alpha_i, \qquad
v = -(x+o_x)\sin\alpha_i + (y+o_y)\cos\alpha_i, \qquad
\text{blocked if } u>0 \ \text{and}\ |v|<w/2 ,
$$

where $(o_x, o_y)$ are the optional per-arm offsets in metres (the arm origin sits at $-(o_x, o_y)$). The arms extend to the pupil edge. The area $A_{eff}$ is recomputed.

### Limitations

- The pupil is the same for all fields and wavelengths (no pupil shift or distortion with field).
- Spiders and obstruction are pixelated; with low resolution thin spiders are quantised to whole pixels.

::: SAOS.Telescope.Telescope
    options:
      show_root_heading: false
      show_root_full_path: false
      members: true
