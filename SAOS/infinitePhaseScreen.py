"""
Infinite Phase Screens
----------------------

An implementation of the "infinite phase screen", as deduced by Francois Assemat and Richard W. Wilson, 2006.
"""


import logging
import logging.handlers
from queue import Queue

import numpy as np
from scipy import linalg
from scipy.spatial.distance import cdist
from scipy.special import gamma, kv

class PhaseScreenVonKarman():
    """
    A "Phase Screen" for use in AO simulation with Von Karmon statistics.

    This represents the phase addition light experiences when passing through atmospheric
    turbulence. Unlike other phase screen generation techniques that translate a large static
    screen, this method keeps a small section of phase, and extends it as necessary for as many
    steps as required. This can significantly reduce memory consumption at the expense of more
    processing power required.

    The technique is described in a paper by Assemat and Wilson, 2006. It essentially assumes that
    there are two matrices, "A" and "B", that can be used to extend an existing phase screen.
    A single row or column of new phase can be represented by

        X = A.Z + B.b

    where X is the new phase vector, Z is some number of columns of the existing screen,
    and b is a random vector with gaussian statistics.

    This object calculates the A and B matrices using an expression of the phase covariance when it
    is initialised. Calculating A is straightforward through the relationship:

        A =  Cov_xz . (Cov_zz)^(-1).

    B is less trivial.

        BB^t = Cov_xx - A.Cov_zx

    (where B^t is the transpose of B) is a symmetric matrix, hence B can be expressed as

        B = UL,

    where U and L are obtained from the svd for BB^t

        U, w, U^t = svd(BB^t)

    L is a diagonal matrix where the diagonal elements are w^(1/2).

    On initialisation an initial phase screen is calculated using an FFT based method.
    When ``add_row`` is called, a new vector of phase is added to the phase screen using `nCols`
    columns of previous phase. Assemat & Wilson claim that two columns are adequate for good
    atmospheric statistics. The phase in the screen data is always accessed as ``<phasescreen>.scrn`` and is in radians.

        .. note::
        The phase screen is returned on each iteration as a 2d array, with each element representing the phase 
        change in **radians**. This means that to obtain the physical phase distortion in nanometres, 
        it must be multiplied by (wavelength / (2*pi)), (where `wavellength` here is the same wavelength
        in which r0 is given in the function arguments)

    Parameters:
        nx_size (int): Size of phase screen (NxN)
        pixel_scale(float): Size of each phase pixel in metres
        r0 (float): fried parameter (metres)
        L0 (float): Outer scale (metres)
        random_seed (int, optional): seed for the random number generator
        n_columns (int, optional): Number of columns to use to continue screen, default is 2
        far_rows (bool, optional): Add to the stencil sparse rows (columns) further back in the screen, at distances 4, 8,
            16, ... px with a stride that grows with the distance (as in the Fried & Clark stencil). With only the adjacent
            rows the large scales are lost along the direction of motion (tilt ~0.7 of the theory along it). Default True.
        subharmonics (bool, optional): Add subharmonics (3 levels, Lane et al. 1992 / Johansson & Gavel 1994) to the
            initial FFT screen. Without them a screen the size of the pupil has ~1/4 of the tilt variance. Default True.
        initial_oversize (float, optional): The initial FFT screen is generated on a grid this times larger and cropped,
            to avoid the periodicity of an FFT screen of the pupil size (distorted focus/astigmatism). Default 4 (1: old).
    """
    def __init__(self, nx_size, pixel_scale, r0, L0, random_seed=None, n_columns=2, from_file=False, screen_file=None, logger=None,
                 far_rows=True, subharmonics=True, initial_oversize=4):

        self.n_columns = n_columns
        self.far_rows = far_rows
        self.subharmonics = subharmonics
        self.initial_oversize = initial_oversize
        self.nx_size = nx_size
        self.pixel_scale = pixel_scale
        self.r0 = r0
        self.L0 = L0
        self.stencil_length_factor = 1
        self.stencil_length = self.nx_size

        self.random_seed = random_seed
        
        # Setup logger
        if logger is None:
            self.queue_listerner = self.setup_logging()
            self.logger = logging.getLogger()
            self.external_logger_flag = False
        else:
            self.external_logger_flag = True
            self.logger = logger

        self.set_coords()
        self.set_stencil_coords()

        if from_file:
            self.load_from_file(scrn=screen_file)
        else:

            # Vertical movement case
            separations = self.calc_separations(self.stencil_positions_vert, self.vert_positions)
            self.cov_mat_xx_vert, self.cov_mat_zz_vert, self.cov_mat_xz_vert, self.cov_mat_zx_vert, self.cov_mat_vert = self.make_covmats(separations, self.n_stencils)

            self.A_vert = self.makeAMatrix(self.cov_mat_zz_vert, self.cov_mat_xz_vert)
            self.B_vert = self.makeBMatrix(self.cov_mat_xx_vert, self.cov_mat_zx_vert, self.A_vert)

            # Horizontal movement case
            separations = self.calc_separations(self.stencil_positions_horz, self.horz_positions)
            self.cov_mat_xx_horz, self.cov_mat_zz_horz, self.cov_mat_xz_horz, self.cov_mat_zx_horz, self.cov_mat_horz = self.make_covmats(separations, self.n_stencils)

            self.A_horz = self.makeAMatrix(self.cov_mat_zz_horz, self.cov_mat_xz_horz)
            self.B_horz = self.makeBMatrix(self.cov_mat_xx_horz, self.cov_mat_zx_horz, self.A_horz)

            self.make_initial_screen()

    def load_from_file(self, scrn):
        if scrn['A_vert'].shape[1] != self.n_stencils:
            # Screen saved with the stencil of the adjacent rows only: keep it so that the stored A/B matrices apply
            self.far_rows = False
            self.set_stencil_coords()
        self._scrn = scrn['phase'].copy()
        self._R = np.random.default_rng(self.random_seed)  
        # Vertical
        self.A_vert = scrn['A_vert'].copy()
        self.B_vert = scrn['B_vert'].copy()
        # Horizontal
        self.A_horz = scrn['A_horz'].copy()
        self.B_horz = scrn['B_horz'].copy()  

    def set_coords(self):
        """
        Sets the coords of the new phase vector.
        """
        # Coordinates for vertical movement
        self.vert_coords = np.zeros((self.nx_size, 2))
        self.vert_coords[:, 0] = -1
        self.vert_coords[:, 1] = np.arange(self.nx_size)
        self.vert_positions = self.vert_coords * self.pixel_scale

        # Coordinates for horizontal movement
        self.horz_coords = np.zeros((self.nx_size, 2))
        self.horz_coords[:, 0] = np.arange(self.nx_size)
        self.horz_coords[:, 1] = -1
        self.horz_positions = self.horz_coords * self.pixel_scale

    def far_stencil_lines(self):
        """(distance in rows, stride across) of the sparse far rows of the stencil."""
        lines = []
        d = max(4, 2 * self.n_columns)
        while self.far_rows and d < self.stencil_length:
            lines.append((d, max(2, d // 2)))
            d *= 2
        if self.far_rows and self.stencil_length - 1 > self.n_columns and (not lines or lines[-1][0] != self.stencil_length - 1):
            lines.append((self.stencil_length - 1, max(2, (self.stencil_length - 1) // 2)))
        return lines

    def set_stencil_coords(self):
        # Vertical --> Top: the n_columns rows next to the new one, plus sparse far rows (every `stride` px, edges included)
        self.stencil_vert = np.zeros((self.stencil_length, self.nx_size))
        self.stencil_vert[:self.n_columns, :] = 1
        for d, stride in self.far_stencil_lines():
            cols = np.unique(np.r_[np.arange(0, self.nx_size, stride), self.nx_size - 1])
            self.stencil_vert[d, cols] = 1

        self.stencil_coords_vert = np.array(np.where(self.stencil_vert==1)).T
        self.stencil_positions_vert = self.stencil_coords_vert * self.pixel_scale

        # Horizontal --> Left

        self.stencil_horz = np.zeros((self.stencil_length, self.nx_size))
        self.stencil_horz[:, :self.n_columns] = 1
        for d, stride in self.far_stencil_lines():
            rows = np.unique(np.r_[np.arange(0, self.nx_size, stride), self.nx_size - 1])
            self.stencil_horz[rows, d] = 1

        self.stencil_coords_horz = np.array(np.where(self.stencil_horz==1)).T
        self.stencil_coords_horz =  self.stencil_coords_horz[np.lexsort((self.stencil_coords_horz[:, 0], self.stencil_coords_horz[:, 1]))]
        self.stencil_positions_horz = self.stencil_coords_horz * self.pixel_scale

        self.n_stencils = len(self.stencil_coords_vert)

    def calc_separations(self, stencil_positions, new_positions):
        """
        Calculates the separations between the phase points in the stencil and the new phase vector
        """
        positions = np.append(stencil_positions, new_positions, axis=0)
        return cdist(positions, positions)

    def make_covmats(self, separations, n_stencils):
        """
        Makes the covariance matrices required for adding new phase
        """
        cov_mat = self.phase_covariance(separations, self.r0, self.L0)

        cov_mat_zz = cov_mat[:n_stencils, :n_stencils]
        cov_mat_xx = cov_mat[n_stencils:, n_stencils:]
        cov_mat_zx = cov_mat[:n_stencils, n_stencils:]
        cov_mat_xz = cov_mat[n_stencils:, :n_stencils]

        return cov_mat_xx, cov_mat_zz, cov_mat_xz, cov_mat_zx, cov_mat

    def phase_covariance(self, r, r0, L0):
        """
        Calculate the phase covariance between two points seperated by `r`, 
        in turbulence with a given `r0 and `L0`.
        Uses equation 5 from Assemat and Wilson, 2006.

        Parameters:
            r (float, ndarray): Seperation between points in metres (can be ndarray)
            r0 (float): Fried parameter of turbulence in metres
            L0 (float): Outer scale of turbulence in metres
        """
        # Make sure everything is a float to avoid nasty surprises in division!
        r = np.asarray(r, dtype=np.float64)
        r0 = float(r0)
        L0 = float(L0)

        # Get rid of any zeros
        r += 1e-40

        A = (L0 / r0) ** (5. / 3)

        B1 = (2 ** (-5. / 6)) * gamma(11. / 6) / (np.pi ** (8. / 3))
        B2 = ((24. / 5) * gamma(6. / 5)) ** (5. / 6)

        C = (((2 * np.pi * r) / L0) ** (5. / 6)) * kv(5. / 6, (2 * np.pi * r) / L0)

        cov = A * B1 * B2 * C

        return cov        

    def makeAMatrix(self, cov_mat_zz, cov_mat_xz):
        """
        Calculates the "A" matrix, that uses the existing data to find a new 
        component of the new phase vector.
        """

        try:
            cf = linalg.cho_factor(cov_mat_zz)
            inv_cov_zz = linalg.cho_solve(cf, np.identity(cov_mat_zz.shape[0]))
            A_mat = cov_mat_xz.dot(inv_cov_zz)
        except linalg.LinAlgError:
            self.logger.warning("PhaseScreenVonKarman::makeAMatrix - Cholesky solve failed. Performing least squares inversion...")
            inv_cov_zz = np.linalg.lstsq(cov_mat_zz, np.identity(cov_mat_zz.shape[0]), rcond=1e-8)
            A_mat = cov_mat_xz.dot(inv_cov_zz[0])


        return A_mat

    def makeBMatrix(self, cov_mat_xx, cov_mat_zx, A_mat):
        """
        Calculates the "B" matrix, that turns a random vector into a component of the new phase.
        """
        # Can make initial BBt matrix first
        BBt = cov_mat_xx - A_mat.dot(cov_mat_zx)

        # Then do SVD to get B matrix
        u, W, _ = np.linalg.svd(BBt)

        L_mat = np.zeros((self.nx_size, self.nx_size))
        np.fill_diagonal(L_mat, np.sqrt(W))

        # Now use sqrt(eigenvalues) to get B matrix
        B_mat = u.dot(L_mat)

        return B_mat

    def make_initial_screen(self):
        """
        Makes the initial screen using FFT method that can be extended 
        """

        # phase screen will make it *really* random if no seed at all given.
        # If a seed is here, screen must be repeatable wiht same seed
        self._R = np.random.default_rng(self.random_seed)

        self._scrn = self.initial_screen(self._R)

    def initial_screen(self, R):
        """
        FFT screen generated on a grid initial_oversize times larger than the screen (and with subharmonics, if enabled),
        cropped at the centre: a periodic FFT screen of the size of the pupil distorts the low orders (astigmatism,
        focus) because of its periodicity.
        """
        N = int(np.ceil(self.initial_oversize * self.stencil_length))
        scrn = self.von_karman_phase_screen(self.r0, N, self.pixel_scale, self.L0, 1e-10, seed=R, subharmonics=self.subharmonics)
        c = (N - self.stencil_length) // 2
        return scrn[c:c + self.stencil_length, c:c + self.nx_size]

    def von_karman_phase_screen(self, r0, N, delta, L0, l0, seed=None, subharmonics=False):
        """
        Creates a random phase screen with Von Karmen statistics.
        (Schmidt 2010)
        
        Parameters:
            r0 (float): r0 parameter of scrn in metres
            N (int): Size of phase scrn in pxls
            delta (float): size in Metres of each pxl
            L0 (float): Size of outer-scale in metres
            l0 (float): inner scale in metres
            seed (int, optional): seed for random number generator. If provided, 
                allows for deterministic screens  

        .. note::
            The phase screen is returned as a 2d array, with each element representing the phase 
            change in **radians**. This means that to obtain the physical phase distortion in nanometres, 
            it must be multiplied by (wavelength / (2*pi)), (where `wavellength` here is the same wavelength
            in which r0 is given in the function arguments)

        Returns:
            ndarray: np array representing phase screen in radians
        """
        delta = float(delta)
        r0 = float(r0)
        L0 = float(L0)
        l0 = float(l0)

        if seed is None:
            R = np.random.default_rng()
        else:
            R = np.random.default_rng(seed)

        del_f = 1./(N*delta)

        fx = np.arange(-N/2., N/2.) * del_f

        (fx, fy) = np.meshgrid(fx,fx)
        f = np.sqrt(fx**2. + fy**2.)

        fm = 5.92/l0/(2*np.pi)
        f0 = 1./L0

        PSD_phi = (0.023*r0**(-5./3.) * np.exp(-1*((f/fm)**2)) / (((f**2) + (f0**2))**(11./6)))

        PSD_phi[int(N/2), int(N/2)] = 0

        cn = ((R.normal(size=(N, N))+1j * R.normal(size=(N, N))) * np.sqrt(PSD_phi)*del_f)

        phs =np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(cn))) * (N) ** 2
        phs = phs.real

        if subharmonics:
            # Scales larger than the screen: 3 levels of 3x3 frequencies with steps del_f/3^p (Lane et al. 1992,
            # Johansson & Gavel 1994), the central one excluded
            x = (np.arange(N) - N / 2) * delta                  # columns (x) and rows (y) share the coordinates
            phs_lo = np.zeros((N, N), dtype=complex)
            for p in range(1, 4):
                del_fp = del_f / 3 ** p
                fx_p = np.arange(-1, 2) * del_fp
                (fx_p, fy_p) = np.meshgrid(fx_p, fx_p)
                f_p = np.sqrt(fx_p ** 2 + fy_p ** 2)
                PSD_p = 0.023 * r0 ** (-5. / 3.) * np.exp(-1 * ((f_p / fm) ** 2)) / ((f_p ** 2 + f0 ** 2) ** (11. / 6))
                PSD_p[1, 1] = 0
                cn_p = (R.normal(size=(3, 3)) + 1j * R.normal(size=(3, 3))) * np.sqrt(PSD_p) * del_fp
                for i in range(3):
                    for j in range(3):
                        # exp(i 2 pi (fx x + fy y)) is separable: outer product of the row and column factors
                        phs_lo += cn_p[i, j] * np.outer(np.exp(1j * 2 * np.pi * fy_p[i, j] * x), np.exp(1j * 2 * np.pi * fx_p[i, j] * x))
            phs = phs + phs_lo.real - phs_lo.real.mean()

        return phs        

    def get_new_row(self, sign=1): # Vertical movement --> + adds row to the bottom, - adds row to the top
        random_data = self._R.normal(0, 1, size=self.nx_size)

        if sign < 0: # add row at the top --> coincides with the coordinates vector
            temp_scrn = self.scrn
        else: # add row at the bottom --> we need to flip the data
            temp_scrn = self.scrn[::-1]

        stencil_data = temp_scrn[(self.stencil_coords_vert[:, 0], self.stencil_coords_vert[:, 1])]


        new_row = self.A_vert.dot(stencil_data) + self.B_vert.dot(random_data)

        new_row.shape = (1, self.nx_size)

        return new_row
    
    def get_new_col(self, sign=1): # Horizontal movement --> + adds col to the left, - adds col to the right
        random_data = self._R.normal(0, 1, size=self.nx_size)

        if sign < 0: # add col to the right --> we need to flip the data
            temp_scrn = self.scrn[:, ::-1]
        else: # add col to the left --> coincides with the coordinates vector
            temp_scrn = self.scrn

        stencil_data = temp_scrn[(self.stencil_coords_horz[:, 0], self.stencil_coords_horz[:, 1])]


        new_col = self.A_horz.dot(stencil_data) + self.B_horz.dot(random_data)

        new_col.shape = (self.nx_size, 1)

        return new_col    

    def add_row(self, sign=1): # Vertical movement
        """
        Adds a new row to the phase screen and removes old ones.
        """

        new_row = self.get_new_row(sign)

        if sign < 0: # add row at the top
            tmp_scrn = np.append(new_row, self._scrn, axis=0)
            self._scrn = tmp_scrn[:self.stencil_length, :self.nx_size]

        else: # add row at the bottom
            tmp_scrn = np.append(self._scrn, new_row, axis=0)
            self._scrn = tmp_scrn[-self.stencil_length:, :self.nx_size]

        return self.scrn

    def add_col(self, sign=1): # Horizontal movement
        """
        Adds a new col to the phase screen and removes old ones.
        """

        new_col = self.get_new_col(sign)

        if sign < 0: # add col to the right
            tmp_scrn = np.append(self._scrn, new_col, axis=1)
            self._scrn = tmp_scrn[:self.nx_size, -self.stencil_length:]

        else: # add col to the left
            tmp_scrn = np.append(new_col, self._scrn, axis=1)
            self._scrn = tmp_scrn[:self.nx_size, :self.stencil_length]

        return self.scrn    

    @property
    def scrn(self):
        """
        The current phase map held in the PhaseScreen object in radians.
        """
        return self._scrn[:self.nx_size, :self.nx_size]
    
    def setup_logging(self, logging_level=logging.WARNING):
        #
        #  Setup of logging at the main process using QueueHandler
        log_queue = Queue()
        queue_handler = logging.handlers.QueueHandler(log_queue)
        root_logger = logging.getLogger()
        root_logger.setLevel(logging_level)  # Minimum log level

        # Setup of the formatting
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s"
        )

        # Output to terminal
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        # Qeue handler captures the messages from the different logs and serialize them
        queue_listener = logging.handlers.QueueListener(log_queue, console_handler)
        root_logger.addHandler(queue_handler)
        queue_listener.start()

        return queue_listener
    
    # The logging Queue requires to stop the listener to avoid having an unfinalized execution. 
    # If the logger is external, then the queue is stop outside of the class scope and we shall
    # avoid to attempt its destruction
    def __del__(self):
        if not self.external_logger_flag:
            self.queue_listerner.stop()