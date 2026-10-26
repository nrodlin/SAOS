"""
T1 validation of the modal bases used to command the DMs and to measure interaction matrices: rank, orthogonality,
piston content, disk-harmonic and Karhunen-Loeve definitions (independent operator / eigenproblem references), Zernike shapes
and sign convention, and accuracy of the Zernike radial polynomials up to the orders
an EST-size DM needs.

Modules under test: SAOS.modalBasis.{zonalModes, discHarmonicModes, karhunenLoeveModes, hadamardModes, zernikeModes}.
Bases are evaluated on the actuator coordinates of a DeformableMirror (or of an equivalent cartesian grid for the
EST-size cases, which only need coordinates); columns are modes, normalised to max |value| = 1.
"""
import types

import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.special import eval_jacobi

from citools import known_issue
from SAOS.DeformableMirror import DeformableMirror
from SAOS.Telescope import Telescope
from SAOS.modalBasis.discHarmonicModes import generate_dh_modes
from SAOS.modalBasis.hadamardModes import generate_hadamard_modes
from SAOS.modalBasis.karhunenLoeveModes import generate_kl_modes
from SAOS.modalBasis.zernikeModes import ZernikeNaive, generate_zernike_modes
from SAOS.modalBasis.zonalModes import generate_zonal_modes

pytestmark = pytest.mark.fast

GENERATORS = dict(zonal=generate_zonal_modes, dh=generate_dh_modes, kl=generate_kl_modes,
                  hadamard=generate_hadamard_modes, zernike=generate_zernike_modes)

NMODES_PISTON = known_issue(
    'MB-NMODES-PISTON',
    'with include_piston=False the default nModes is nValidAct, but only nValidAct-1 piston-free modes exist: '
    'Hadamard loses one rank, DH/KL put piston into the last modes (QR)',
    'discHarmonicModes.py:8-13, karhunenLoeveModes.py:14-18, hadamardModes.py:5-8')


@pytest.fixture(scope='module')
def dm11(logger):
    """11x11 cartesian ground DM (97 valid actuators) on a 32 px pupil: only its actuator geometry is used."""
    tel = Telescope(resolution=32, diameter=4.0, samplingTime=1 / 2000, fov=0, logger=logger)
    return DeformableMirror(telescope=tel, nActs=11, altitude=0, logger=logger, device='cpu', validActThreshpercentage=0.5)


def cartesian_actuators(n, thresh_pitch, D=1.0):
    """Actuator geometry of DeformableMirror.generate_cartesian_dm (no influence functions): n x n grid over D."""
    x = np.linspace(-D / 2, D / 2, n)
    X, Y = np.meshgrid(x, x)
    coords = np.array([X.ravel(), Y.ravel()]).T
    valid = np.hypot(coords[:, 0], coords[:, 1]) <= D / 2 + thresh_pitch * D / (n - 1)
    return types.SimpleNamespace(coordinates=coords, validAct=valid, nValidAct=int(valid.sum()), altitude=0)


def radial_reference(n, m, rho):
    """Zernike radial polynomial through scipy's Jacobi polynomials: (-1)^k rho^m P_k^(m,0)(1 - 2 rho^2), k = (n-m)/2."""
    k = (n - m) // 2
    return (-1) ** k * rho ** m * eval_jacobi(k, m, 0, 1 - 2 * rho ** 2)


def piston_fraction(B):
    """|mean| / rms of every mode over the valid actuators."""
    return np.abs(B.mean(axis=0)) / np.sqrt((B ** 2).mean(axis=0))


# ----------------------------------------------------------------------------------------------------------------------
# Rank, orthogonality and piston
# ----------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize('name', ['zonal', pytest.param('dh', marks=NMODES_PISTON), pytest.param('kl', marks=NMODES_PISTON),
                                  pytest.param('hadamard', marks=NMODES_PISTON)])
def test_basis_spans_actuator_space(name, dm11, report, tolerances):
    """
    A modal basis must be a basis of the commandable space: rank = number of modes (singular values > 1e-10 s_max).
    Piston-free bases (dh, kl, hadamard, include_piston=False) must contain no piston (|mean|/rms < 1e-10 per mode) and
    therefore at most nValidAct-1 modes. The zonal basis is the identity on the valid actuators.
    """
    tol = tolerances['modal_basis']['basis']
    B = GENERATORS[name](dm11)[dm11.validAct]
    s = np.linalg.svd(B, compute_uv=False)
    rank = int((s > tol['rank_rel_sv'] * s[0]).sum())
    report.metric('valid actuators', dm11.nValidAct)
    report.check('rank - number of modes', rank - B.shape[1], expected=0, atol=0)
    if name != 'zonal':
        pf = piston_fraction(B)
        report.check('max piston fraction (|mean|/rms)', pf.max(), hi=tol['piston_fraction'])
        report.check('number of modes <= nValidAct - 1', B.shape[1], hi=dm11.nValidAct - 1)
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.semilogy(s / s[0], '.-')
    ax.set_xlabel('index'); ax.set_ylabel('singular value / max')
    ax.set_title(f'{name} basis on 11x11 DM ({dm11.nValidAct} valid actuators): rank {rank} of {B.shape[1]}', fontsize=9)
    ax.grid(True, which='both', color='0.9')
    report.figure(fig, f'singular_values_{name}')
    report.assert_ok()


@pytest.mark.parametrize('name', ['zonal', 'dh', 'kl'])
def test_orthogonal_bases_have_diagonal_gram(name, dm11, report, tolerances):
    """Orthonormalised bases (zonal, DH and KL by QR) have a diagonal Gram matrix on the valid actuators."""
    B = GENERATORS[name](dm11)[dm11.validAct]
    G = B.T @ B
    off = np.max(np.abs(G - np.diag(np.diag(G)))) / np.max(np.diag(G))
    report.check('max |off-diagonal Gram| / max diagonal', off, hi=tolerances['modal_basis']['basis']['gram_offdiag'])
    report.assert_ok()


def test_zernike_rank_limited_by_actuator_sampling(dm11, report, tolerances):
    """
    Zernike polynomials sampled on the actuator grid are linearly independent only up to the order the grid resolves:
    the first half of the modes must be full rank. With nModes = nValidAct (the default) the high orders alias and the
    basis is rank deficient; this is a property of the sampling, not of the polynomial evaluation (a Jacobi-based
    evaluation gives the same rank), so it is recorded and not failed: IMs measured with the full Zernike basis contain
    dependent columns that only the regularisation removes.
    """
    tol = tolerances['modal_basis']['basis']
    for label, dm in (('11x11', dm11), ('37x37 (EST ASM size)', cartesian_actuators(37, 0.7533))):
        B = generate_zernike_modes(dm)[dm.validAct]
        half = B.shape[1] // 2
        s_half = np.linalg.svd(B[:, :half], compute_uv=False)
        s_all = np.linalg.svd(B, compute_uv=False)
        report.check(f'{label}: rank of the first {half} modes - {half}', int((s_half > tol['rank_rel_sv'] * s_half[0]).sum()) - half,
                     expected=0, atol=0)
        report.metric(f'{label}: rank with nModes = nValidAct = {B.shape[1]}', int((s_all > tol['rank_rel_sv'] * s_all[0]).sum()))
        report.check(f'{label}: max piston fraction', piston_fraction(B).max(), hi=tol['piston_fraction'])
    report.assert_ok()


# ----------------------------------------------------------------------------------------------------------------------
# Disk harmonics and Karhunen-Loeve: independent operator / eigenproblem definitions
# ----------------------------------------------------------------------------------------------------------------------
#
# Independent references written here from the published definitions (no SAOS code):
#   * Noll ordering j -> (n, m, cos|sin): shells n = 0, 1, ...; inside a shell increasing |m|; a pair (+-m) takes two
#     consecutive j, the even j being the cosine term (Noll 1976).
#   * Zernike polynomials Z = N R_n^|m|(rho) {cos, sin}(m theta), N = sqrt(n+1) (m = 0), sqrt(2 (n+1)) (m != 0), R through
#     Jacobi polynomials; rho = r / (radius of the outermost valid actuator).
#   * Kolmogorov covariance of Noll-normalised Zernike coefficients (Noll 1976 eq. 16), in (D/r0)^(5/3) units.
#   * Disk harmonics (Milton & Lloyd-Hart 2005): Neumann eigenfunctions of the Laplacian on the disk, J_m(k r) {cos, sin}(m theta)
#     with J_m'(k) = 0, ordered by eigenvalue k.

def noll_table(jmax):
    """List of (j, n, m, trig) for j = 1..jmax; trig 'cos' / 'sin' / None (m = 0)."""
    out, j, n = [], 1, 0
    while j <= jmax:
        ms = sorted(range(n % 2, n + 1, 2))
        for m in ms:
            if m == 0:
                out.append((j, n, 0, None)); j += 1
            else:
                for jj in (j, j + 1):
                    out.append((jj, n, m, 'cos' if jj % 2 == 0 else 'sin'))
                j += 2
        n += 1
    return [o for o in out if o[0] <= jmax]


def zernike_values(j, x, y):
    """Noll-normalised Zernike j on the unit disk at (x, y) (r > 1 not expected)."""
    _, n, m, trig = noll_table(j)[j - 1]
    rho, th = np.hypot(x, y), np.arctan2(y, x)
    R = radial_reference(n, m, rho)
    if trig is None:
        return np.sqrt(n + 1) * R
    return np.sqrt(2 * (n + 1)) * R * (np.cos(m * th) if trig == 'cos' else np.sin(m * th))


def noll_covariance(js):
    """Kolmogorov covariance <a_j a_j'> of Noll-normalised Zernike coefficients (units (D/r0)^(5/3)), Noll 1976 eq. 16."""
    from scipy.special import gamma
    tab = noll_table(max(js))
    k0 = gamma(14 / 3) * ((24 / 5) * gamma(6 / 5)) ** (5 / 6) * gamma(11 / 6) ** 2 / (2 * np.pi ** 2)
    C = np.zeros((len(js), len(js)))
    for a, ja in enumerate(js):
        _, n, m, _ = tab[ja - 1]
        for b, jb in enumerate(js):
            _, n2, m2, _ = tab[jb - 1]
            if m != m2 or not (m == 0 or ja % 2 == jb % 2):
                continue
            C[a, b] = (k0 * (-1) ** ((n + n2 - 2 * m) // 2) * np.sqrt((n + 1) * (n2 + 1)) * gamma((n + n2 - 5 / 3) / 2)
                       / (gamma((n - n2 + 17 / 3) / 2) * gamma((n2 - n + 17 / 3) / 2) * gamma((n + n2 + 23 / 3) / 2)))
    return C


NOLL_DELTA = [1.0299, 0.582, 0.134, 0.111, 0.0880, 0.0648, 0.0587, 0.0525, 0.0463, 0.0401, 0.0377]   # Noll 1976 table IV, J = 1..11


def actuator_geometry(n, thresh_pitch=0.5, D=1.0):
    """x, y, validity of the n x n actuator grid (row-major) spanning D, valid if r <= D/2 + thresh pitch; own implementation."""
    x = np.linspace(-D / 2, D / 2, n)
    X, Y = np.meshgrid(x, x)
    X, Y = X.ravel(), Y.ravel()
    return X, Y, np.hypot(X, Y) <= D / 2 + thresh_pitch * D / (n - 1)


def unit_disk_coordinates(X, Y, valid):
    r = np.hypot(X[valid], Y[valid]).max()
    return X[valid] / r, Y[valid] / r


def projector(A, rcond=1e-10):
    """Orthogonal projector onto the column space of A."""
    U, s, _ = np.linalg.svd(A, full_matrices=False)
    return U[:, s > rcond * s[0]] @ U[:, s > rcond * s[0]].T


def subspace_distance(A, B):
    """Spectral norm of the difference of the projectors = sin of the largest principal angle."""
    return np.linalg.norm(projector(A) - projector(B), 2)


def test_noll_covariance_reference_matches_noll_table(report):
    """
    Sanity of the independent covariance: variances of Z_2..Z_11 relative to that of Z_2 equal (Delta_{j-1} - Delta_j) /
    (Delta_1 - Delta_2) of Noll's table (4 digits; absolute 1e-3 covers the table rounding (4 digits in the differences of Delta)). The overall constant of the
    published formula is 1.3 % above the table (0.4536 vs 0.4479 for Z_2), irrelevant for eigenvectors (recorded).
    """
    js = np.arange(2, 12)
    var = np.diag(noll_covariance(list(js)))
    table = np.array([NOLL_DELTA[j - 2] - NOLL_DELTA[j - 1] for j in js])
    report.metric('formula / table for Z_2 (overall constant)', var[0] / table[0])
    report.check('variance(Z_j) / variance(Z_2), j = 2..11', var / var[0], table / table[0], atol=1e-3)
    report.assert_ok()


@known_issue('KL-COV-SCALE', 'karhunenLoeveModes applies the Noll covariance matrix (Noll-normalised coefficients) to Zernike columns '
             'that get_zernikes rescales to unit rms on the actuators (rms 0.99-1.22 on an 11x11 grid) and piston-removes: the '
             'diagonalisation is exact for the wrong covariance, the KL modes are not uncorrelated under the atmosphere '
             '(variance deviations up to 43 % for some of the first 20 modes)', 'karhunenLoeveModes.py:25-37, zernikeModes.py:49-59')
def test_kl_modes_are_the_atmospheric_eigenmodes_of_the_zernike_subspace(report, tolerances):
    """
    Karhunen-Loeve modes of an 11x11 DM (first 20 Zernike modes, Noll 2..21 = radial orders 1..5 complete): the span of
    the modes is the span of the piston-free Zernike polynomials sampled on the valid actuators; inside it the modes
    are orthogonal on the actuators and the amplitudes of an atmospheric realisation on them are uncorrelated (double
    diagonalisation): with Z = Q R (independent Noll-normalised Zernike on the actuators, QR), the modes are Q U, U the
    eigenvectors of R C R^T (C: Noll covariance), ordered by decreasing variance. Checked: (1) containment in the Zernike span; (2) variance of every mode a^T C a equals the
    eigenvalue of the same rank, non-increasing; (3) every cluster of degenerate eigenvalues (relative gap 1e-6) spans
    the same subspace as the independent eigenvectors; (4) orthogonality on the actuators and covariance-diagonality.
    """
    tol = tolerances['modal_basis']['kl']
    nm = 20
    X, Y, valid = actuator_geometry(11)
    geo = types.SimpleNamespace(coordinates=np.c_[X, Y], validAct=valid, nValidAct=int(valid.sum()), altitude=0)
    kl = generate_kl_modes(geo, nModes=nm)[valid]
    x, y = unit_disk_coordinates(X, Y, valid)
    Z = np.stack([zernike_values(j, x, y) for j in range(2, 2 + nm)], axis=1)
    Z = Z - Z.mean(axis=0)                                                # piston-free on the actuators
    C = noll_covariance(list(range(2, 2 + nm)))
    # (1) containment
    coef = np.linalg.lstsq(Z, kl, rcond=None)[0]
    report.check('KL modes inside the Zernike span: max residual / max |mode|', np.max(np.abs(kl - Z @ coef)) / np.max(np.abs(kl)), hi=tol['span'])
    # independent KL: orthonormalise Z on the actuators (Z = Q R), diagonalise the covariance of the coefficients b = R a
    Q, Rr = np.linalg.qr(Z)
    lam, U = np.linalg.eigh(Rr @ C @ Rr.T)
    order = np.argsort(lam)[::-1]
    lam, U = lam[order], U[:, order]
    ref = Q @ U
    klu = kl / np.linalg.norm(kl, axis=0)                                  # unit norm on the actuators
    W = Z.T @ klu                                                          # amplitude beta_k = klu_k . phi, phi = Z a  ->  beta = W^T a
    Ck = W.T @ C @ W                                                       # covariance of the KL amplitudes
    var = np.diag(Ck)
    report.check('variance of SAOS mode k / eigenvalue k - 1', var / lam - 1, expected=0, atol=tol['eig_rel'])
    report.check('variances non-increasing (max var[k+1] - var[k]) / var[0]', np.max(np.diff(var)) / var[0], hi=tol['eig_rel'])
    # (3) degenerate clusters
    edges = np.flatnonzero(np.abs(np.diff(lam)) > 1e-6 * lam[0]) + 1
    clusters = np.split(np.arange(nm), edges)
    worst = 0.0
    for cl in clusters:
        worst = max(worst, subspace_distance(kl[:, cl], ref[:, cl]))
    report.metric('number of eigenvalue clusters', len(clusters))
    report.check('clusters: max subspace distance SAOS vs independent eigenvectors', worst, hi=tol['subspace'])
    # (4) orthogonality
    G = klu.T @ klu
    report.check('orthogonality on actuators: max off-diagonal of unit-norm Gram', np.max(np.abs(G - np.eye(nm))), hi=tol['subspace'])
    report.check('covariance diagonality of the KL amplitudes: max |off-diagonal| / max diagonal', np.max(np.abs(Ck - np.diag(var))) / var.max(), hi=tol['eig_rel'])
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.semilogy(np.arange(1, nm + 1), lam, 'k-', label='independent eigenvalues')
    ax.semilogy(np.arange(1, nm + 1), var, 'o', mfc='none', label='variance of SAOS modes')
    ax.set_xlabel('KL mode'); ax.set_ylabel('variance [(D/r0)^(5/3) rad^2]'); ax.legend(fontsize=8, frameon=False)
    report.figure(fig, 'kl_variances')
    report.assert_ok()


def disk_harmonic_functions(x, y, kmax_count=40):
    """
    Independent Neumann eigenfunctions of the Laplacian on the unit disk evaluated at (x, y), sorted by eigenvalue k:
    returns (k list, columns). m = 0: constant excluded, roots of J_0'; m > 0: cos and sin of J_m(k r), J_m'(k) = 0.
    """
    from scipy.special import jnp_zeros, jv
    items = []
    for m in range(0, 12):
        for k in jnp_zeros(m, 8):
            items.append((k, m, 'cos'))
            if m > 0:
                items.append((k, m, 'sin'))
    items.sort(key=lambda t: (t[0], t[1], t[2]))
    r, th = np.hypot(x, y), np.arctan2(y, x)
    cols, ks = [], []
    for k, m, trig in items[:kmax_count]:
        f = jv(m, k * r) * (1 if m == 0 else (np.cos(m * th) if trig == 'cos' else np.sin(m * th)))
        cols.append(f - f.mean()); ks.append(k)
    return np.array(ks), np.stack(cols, axis=1)


@pytest.mark.parametrize('count', [2, 5])
def test_dh_lowest_modes_span_the_lowest_neumann_eigenfunctions(count, report, tolerances):
    """
    Disk harmonics are the eigenfunctions of the Laplacian on the disk with zero normal derivative at the edge
    (Milton & Lloyd-Hart): J_m(k r) {cos, sin}(m theta), J_m'(k) = 0. The first `count` SAOS modes (count = 2: k = 1.841 pair;
    5: + the 3.054 pair and the k = 3.832 axisymmetric mode; complete eigenvalue levels) span the `count` lowest eigenfunctions
    of the independent construction, evaluated on the valid actuators of an 11x11 DM (disk radius = outermost valid
    actuator, piston removed on the actuators): spectral norm of the difference of the projectors < yaml bound.
    """
    X, Y, valid = actuator_geometry(11)
    geo = types.SimpleNamespace(coordinates=np.c_[X, Y], validAct=valid, nValidAct=int(valid.sum()), altitude=0)
    dh = generate_dh_modes(geo, nModes=count)[valid]
    x, y = unit_disk_coordinates(X, Y, valid)
    ks, F = disk_harmonic_functions(x, y)
    report.metric('independent eigenvalues k used', ks[:count])
    report.check(f'first {count} modes: subspace distance to the {count} lowest Neumann eigenfunctions', subspace_distance(dh, F[:, :count]),
                 hi=tolerances['modal_basis']['dh']['subspace'])
    report.assert_ok()


@pytest.mark.parametrize('nmax', [1, 2, 3, 4])
def test_dh_radial_order_shells_span_documented_functions(nmax, report, tolerances):
    """
    Documented labelling of the SAOS disk harmonics (Noll-like radial order n, azimuthal order m; the mode of order
    (n, m) is J_m(k r) {cos, sin}(m theta) with k the q-th root of J_m', q = (n - |m|)/2 + 1 for m != 0 and n/2 for m = 0):
    the first modes that complete radial orders 1..nmax span exactly these functions (independent evaluation).
    Informational: distance of the same span to the lowest-eigenvalue set (the labelling is not eigenvalue-ordered).
    """
    from scipy.special import jnp_zeros, jv
    X, Y, valid = actuator_geometry(11)
    geo = types.SimpleNamespace(coordinates=np.c_[X, Y], validAct=valid, nValidAct=int(valid.sum()), altitude=0)
    x, y = unit_disk_coordinates(X, Y, valid)
    r, th = np.hypot(x, y), np.arctan2(y, x)
    cols = []
    for n in range(1, nmax + 1):
        for m in range(n % 2, n + 1, 2):
            q = n // 2 if m == 0 else (n - m) // 2 + 1
            k = jnp_zeros(m, q)[-1]
            for trig in ((None,) if m == 0 else ('cos', 'sin')):
                f = jv(m, k * r) * (1 if trig is None else (np.cos(m * th) if trig == 'cos' else np.sin(m * th)))
                cols.append(f - f.mean())
    F = np.stack(cols, axis=1)
    dh = generate_dh_modes(geo, nModes=F.shape[1])[valid]
    report.metric('modes in the shells', F.shape[1])
    report.check(f'radial orders 1..{nmax}: subspace distance SAOS vs independent functions', subspace_distance(dh, F),
                 hi=tolerances['modal_basis']['dh']['subspace'])
    ks, G = disk_harmonic_functions(x, y)
    report.metric('distance of the same span to the lowest-eigenvalue set (information)', subspace_distance(F, G[:, :F.shape[1]]))
    report.assert_ok()


# ----------------------------------------------------------------------------------------------------------------------
# Zernike shapes, sign convention and radial polynomials
# ----------------------------------------------------------------------------------------------------------------------

def _zernike_vs_function(dm, noll, func):
    """Correlation of the Zernike mode with Noll index `noll` (column noll-2) with func(x, y) - mean, on valid actuators."""
    B = generate_zernike_modes(dm, nModes=noll - 1)[dm.validAct]
    x, y = dm.coordinates[dm.validAct].T
    R = np.max(np.hypot(x, y))
    f = func(x / R, y / R)
    f = f - f.mean()
    z = B[:, noll - 2]
    return float(z @ f / np.linalg.norm(z) / np.linalg.norm(f))


@pytest.mark.parametrize('noll,label,func', [(2, 'tip 2 rho cos', lambda x, y: x),
                                             (4, 'defocus 2 rho^2 - 1', lambda x, y: 2 * (x ** 2 + y ** 2) - 1),
                                             (6, 'astigmatism rho^2 cos 2theta', lambda x, y: x ** 2 - y ** 2),
                                             (11, 'spherical 6 rho^4 - 6 rho^2 + 1', lambda x, y: 6 * (x ** 2 + y ** 2) ** 2 - 6 * (x ** 2 + y ** 2) + 1)])
def test_zernike_low_order_shapes(noll, label, func, dm11, report, tolerances):
    """Even-Noll (cosine) and m = 0 Zernike modes equal their analytic shape with positive sign (correlation > 0.999)."""
    report.check(f'Noll {noll} ({label}): correlation', _zernike_vs_function(dm11, noll, func),
                 lo=tolerances['modal_basis']['zernike']['shape_correlation'])
    report.assert_ok()


@known_issue('MB-NOLL-SIGN', 'odd-Noll (sine) Zernike modes have the opposite sign of Noll\'s convention: '
             'sin(m theta) is evaluated with m < 0', 'zernikeModes.py:197-203, 357-359')
@pytest.mark.parametrize('noll,label,func', [(3, 'tilt 2 rho sin', lambda x, y: y),
                                             (5, 'astigmatism rho^2 sin 2theta', lambda x, y: 2 * x * y)])
def test_zernike_noll_sign_convention(noll, label, func, dm11, report, tolerances):
    """Noll's convention: Z3 = 2 rho sin(theta) (+y), Z5 = sqrt(6) rho^2 sin(2 theta) (+xy): correlation > 0.999."""
    report.check(f'Noll {noll} ({label}): correlation', _zernike_vs_function(dm11, noll, func),
                 lo=tolerances['modal_basis']['zernike']['shape_correlation'])
    report.assert_ok()


def _radial_errors(evaluator, n_values, rho):
    out = []
    for n in n_values:
        err = 0.0
        for m in range(n % 2, n + 1, 2):
            err = max(err, np.max(np.abs(evaluator(np.int64(n), np.int64(m), rho.copy()) - radial_reference(n, m, rho))))
        out.append(err)
    return np.array(out)


def test_zernike_radial_polynomial_low_orders(report, tolerances):
    """SAOS radial polynomials (naive factorial sum, used to build the basis) match the Jacobi form for n <= 30."""
    rho = np.linspace(0, 1, 501)
    err = _radial_errors(ZernikeNaive(mask=[]).R_nm, range(0, 31), rho)
    report.check('max |R_naive - R_ref|, n <= 30', err.max(), hi=tolerances['modal_basis']['zernike']['radial_abs_low_orders'])
    report.assert_ok()


def test_zernike_radial_jacobi_evaluator(report, tolerances):
    """The Jacobi evaluator already present in SAOS (R_nm_Jacobi) is accurate up to n = 49 (the fix for MB-ZERN-RADIAL)."""
    rho = np.linspace(0, 1, 501)
    err = _radial_errors(ZernikeNaive(mask=[]).R_nm_Jacobi, range(0, 50, 7), rho)
    report.check('max |R_jacobi - R_ref|, n <= 49', err.max(), hi=tolerances['modal_basis']['zernike']['radial_abs'])
    report.assert_ok()


@known_issue('MB-ZERN-RADIAL', 'the basis uses the naive factorial sum for the radial polynomial, which loses all '
             'precision above n ~ 40 (EST ASM with nModes = nValidAct needs n = 49)', 'zernikeModes.py:207-222, 50')
def test_zernike_radial_polynomial_est_orders(report, tolerances):
    """
    An EST-size DM (ASM, 37x37, Zernike up to n ~ 49) needs radial polynomials accurate to |R| <= 1 up to n = 49. The
    basis that SAOS generates (generate_zernike_modes) must equal, column by column, the same basis built with the
    independent radial reference of this module (Jacobi form, scipy). The error of the naive factorial sum (still present
    in ZernikeNaive.R_nm, no longer used by the basis) is recorded for information.
    """
    tol = tolerances['modal_basis']['zernike']
    rho = np.linspace(0, 1, 501)
    n_values = np.arange(31, 50)
    err = _radial_errors(ZernikeNaive(mask=[]).R_nm, n_values, rho)
    report.metric('max |R_naive - R_ref|, 31 <= n <= 49 (naive sum, unused by the basis)', err.max())
    report.metric('first n with naive error > 1e-3', int(n_values[np.argmax(err > 1e-3)]) if np.any(err > 1e-3) else -1)

    dm = cartesian_actuators(37, 0.7533)
    B_saos = generate_zernike_modes(dm)[dm.validAct]
    orig = ZernikeNaive.R_nm_Jacobi
    try:
        ZernikeNaive.R_nm_Jacobi = lambda self, n, m, r: radial_reference(int(abs(n)), int(abs(m)), r)
        B_ref = generate_zernike_modes(dm)[dm.validAct]
    finally:
        ZernikeNaive.R_nm_Jacobi = orig
    bad = np.max(np.abs(B_saos - B_ref), axis=0) > tol['basis_column_abs']
    report.check(f'37x37 basis ({dm.nValidAct} modes): columns differing from the reference > {tol["basis_column_abs"]}',
                 int(bad.sum()), expected=0, atol=0)
    s = np.linalg.svd(B_saos, compute_uv=False)
    # Information: even exact Zernikes alias on the actuator grid at the highest orders (physical limit, not this bug)
    report.metric('37x37 basis numerical rank (s > 1e-8 s_max) / number of modes', int(np.sum(s > 1e-8 * s[0])) / B_saos.shape[1])
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.semilogy(n_values, np.maximum(err, 1e-17), 'o-')
    ax.axhline(tol['radial_abs'], color='k', lw=0.7)
    ax.set_xlabel('radial order n'); ax.set_ylabel('max |R_naive - R_ref| (|R| <= 1)')
    ax.set_title('Zernike radial polynomial, naive factorial sum (unused by the basis)', fontsize=9)
    ax.grid(True, which='both', color='0.9')
    report.figure(fig, 'radial_error_vs_order')
    report.assert_ok()
