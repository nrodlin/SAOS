"""
Infinite Phase Screens
----------------------

An implementation of the "infinite phase screen", as deduced by Francois Assemat and Richard W. Wilson, 2006.
"""

import numpy as np
from scipy import linalg
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
    """
    def __init__(self, nx_size, pixel_scale, r0, L0, random_seed=None, n_columns=2):

        self.n_columns = n_columns
        self.nx_size = nx_size
        self.pixel_scale = pixel_scale
        self.r0 = r0
        self.L0 = L0
        self.stencil_length_factor = 1
        self.stencil_length = self.nx_size

        self.n_stencils = self.n_columns * self.nx_size

        self.random_seed = random_seed

        self.set_coords()
        self.set_stencil_coords()

        # Vertical movement case
        separations = self.calc_separations(self.stencil_positions_vert, self.vert_positions)
        self.cov_mat_xx_vert, self.cov_mat_zz_vert, self.cov_mat_xz_vert, self.cov_mat_zx_vert = self.make_covmats(separations, self.n_stencils)

        self.A_vert = self.makeAMatrix(self.cov_mat_zz_vert, self.cov_mat_xz_vert)
        self.B_vert = self.makeBMatrix(self.cov_mat_xx_vert, self.cov_mat_zx_vert, self.A_vert)

        # Horizontal movement case
        # separations = self.calc_separations(self.stencil_coords_horz, self.horz_coords)
        # self.cov_mat_xx_horz, self.cov_mat_zz_horz, self.cov_mat_xz_horz, self.cov_mat_zx_horz = self.make_covmats(separations, self.n_stencils)

        # self.A_horz = self.makeAMatrix(self.cov_mat_zz_horz, self.cov_mat_xz_horz)
        # self.B_horz = self.makeBMatrix(self.cov_mat_xx_horz, self.cov_mat_zx_horz, self.A_horz)

        self.make_initial_screen()

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
        # self.horz_coords = np.zeros((self.nx_size, 2))
        # self.horz_coords[:, 0] = np.arange(self.nx_size)
        # self.horz_coords[:, 1] = -1
        # self.horz_positions = self.horz_coords * self.pixel_scale

    def set_stencil_coords(self):
        # Vertical --> Top
        self.stencil_vert = np.zeros((self.stencil_length, self.nx_size))
        self.stencil_vert[:self.n_columns, :] = 1

        self.stencil_coords_vert = np.array(np.where(self.stencil_vert==1)).T
        self.stencil_positions_vert = self.stencil_coords_vert * self.pixel_scale

        # Horizontal --> Left

        # self.stencil_horz = np.zeros((self.stencil_length, self.nx_size))
        # self.stencil_horz[:, :self.n_columns] = 1

        # self.stencil_coords_horz = np.array(np.where(self.stencil_horz==1)).T
        # self.stencil_positions_horz = self.stencil_coords_horz * self.pixel_scale

    def calc_separations(self, stencil_positions, new_positions):
        """
        Calculates the separations between the phase points in the stencil and the new phase vector
        """
        positions = np.append(stencil_positions, new_positions, axis=0)
        separations = np.zeros((len(positions), len(positions)))

        for i, (x1, y1) in enumerate(positions):
            for j, (x2, y2) in enumerate(positions):
                delta_x = x2 - x1
                delta_y = y2 - y1

                delta_r = np.sqrt(delta_x ** 2 + delta_y ** 2)

                separations[i, j] = delta_r

        return separations

    def make_covmats(self, separations, n_stencils):
        """
        Makes the covariance matrices required for adding new phase
        """
        self.cov_mat = self.phase_covariance(separations, self.r0, self.L0)

        cov_mat_zz = self.cov_mat[:n_stencils, :n_stencils]
        cov_mat_xx = self.cov_mat[n_stencils:, n_stencils:]
        cov_mat_zx = self.cov_mat[:n_stencils, n_stencils:]
        cov_mat_xz = self.cov_mat[n_stencils:, :n_stencils]

        return cov_mat_xx, cov_mat_zz, cov_mat_xz, cov_mat_zx

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
        r = np.float32(r)
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
            print("Cholesky solve failed. Performing least squares inversion...")
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
        Makes the initial screen usign FFT method that can be extended 
        """

        # phase screen will make it *really* random if no seed at all given.
        # If a seed is here, screen must be repeatable wiht same seed
        self._R = np.random.default_rng(self.random_seed)

        self._scrn = self.ft_phase_screen(
            self.r0, self.stencil_length, self.pixel_scale, self.L0, 1e-10, seed=self._R
        )

        self._scrn = self._scrn[:, :self.nx_size]
    
    def ft_phase_screen(self, r0, N, delta, L0, l0, FFT=None, seed=None):
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

        return phs


    def get_new_row(self, sign=1): # Vertical movement
        random_data = self._R.normal(0, 1, size=self.nx_size)

        if sign < 0: # add row at the top --> coincides with the coordinates vector
            temp_scrn = self.scrn
        else: # add row at the bottom --> we need to flip the data
            temp_scrn = self.scrn[::-1]

        stencil_data = temp_scrn[(self.stencil_coords_vert[:, 0], self.stencil_coords_vert[:, 1])]


        new_row = self.A_vert.dot(stencil_data) + self.B_vert.dot(random_data)

        new_row.shape = (1, self.nx_size)

        return new_row

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

    @property
    def scrn(self):
        """
        The current phase map held in the PhaseScreen object in radians.
        """
        return self._scrn[:self.nx_size, :self.nx_size]
