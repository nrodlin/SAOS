import numpy as np
from scipy.special import gamma


class StructureFunctionVK:
    """
    Von Karman phase structure function.

    This implementation follows the finite outer-scale expansion used in
    pseudo-analytical AO covariance calculations.
    """

    def __init__(self, r0: float, L0: float, n_max: int = 10, device: str | None = None, use_interpolation: bool = True, max_sep: float = 100.0, n_pts: int = 100000):
        self.r0 = r0
        self.L0 = L0
        self.n_max = n_max
        self.device = device
        self.use_interpolation = use_interpolation

        self.k1 = (
            gamma(11 / 6)
            * 2 ** (1 / 6)
            * np.pi ** (-8 / 3)
            * ((24 / 5) * gamma(6 / 5)) ** (5 / 6)
        )

        a_np = np.zeros(n_max + 1)
        b_np = np.zeros(n_max + 1)

        a_np[0] = gamma(-5 / 6) / 2 ** (1 / 6)
        b_np[0] = gamma(5 / 6) / 2 ** (1 / 6)

        for n in range(1, n_max + 1):
            a_np[n] = a_np[n - 1] / (n * (n + 5 / 6))
            b_np[n] = b_np[n - 1] / (n * (n - 5 / 6))

        self.prefactor = (
            -self.k1
            * self.r0 ** (-5 / 3)
            * self.L0 ** (5 / 3)
        )

        self.a = a_np
        self.b = b_np

        if self.use_interpolation:
            self.max_sep = max_sep
            self.n_pts = n_pts
            self.sep_step = max_sep / (n_pts - 1)
            sep_arr = np.linspace(0, max_sep, n_pts)
            val_arr = self._compute_exact_numpy(sep_arr)
            
            if self.device is not None:
                import torch
                self.val_table = torch.tensor(val_arr, dtype=torch.float32, device=self.device)
            else:
                self.val_table = val_arr.astype(np.float32)
        else:
            if self.device is not None:
                import torch
                self.a = torch.tensor(a_np, dtype=torch.float32, device=self.device)
                self.b = torch.tensor(b_np, dtype=torch.float32, device=self.device)
                self.prefactor_t = torch.tensor(self.prefactor, dtype=torch.float32, device=self.device)
                self.L0_t = torch.tensor(self.L0, dtype=torch.float32, device=self.device)

    def _compute_exact_numpy(self, separation):
        separation = np.asarray(separation, dtype=float)
        x = np.pi * separation / self.L0
        y = x ** (5 / 3)
        x1 = x ** 2

        series = np.zeros_like(separation)
        xn = np.ones_like(separation)
        temp = np.empty_like(separation)

        for n in range(1, self.n_max + 1):
            xn *= x1
            np.multiply(y, self.a[n], out=temp)
            temp += self.b[n]
            temp *= xn
            series += temp

        np.multiply(y, self.a[0], out=temp)
        temp += series
        temp *= self.prefactor

        return temp

    def __call__(self, separation):
        if self.use_interpolation:
            if self.device is not None:
                import torch
                if not isinstance(separation, torch.Tensor):
                    separation = torch.tensor(separation, dtype=torch.float32, device=self.device)
                elif separation.device != torch.device(self.device):
                    separation = separation.to(self.device)

                idx_float = separation / self.sep_step
                idx_lower = torch.floor(idx_float).long()
                idx_lower = torch.clamp(idx_lower, 0, self.n_pts - 2)
                t = idx_float - idx_lower

                val_lower = self.val_table[idx_lower]
                val_upper = self.val_table[idx_lower + 1]

                return val_lower + t * (val_upper - val_lower)
            else:
                separation = np.asarray(separation, dtype=np.float32)
                idx_float = separation / self.sep_step
                idx_lower = np.floor(idx_float).astype(int)
                idx_lower = np.clip(idx_lower, 0, self.n_pts - 2)
                t = idx_float - idx_lower

                val_lower = self.val_table[idx_lower]
                val_upper = self.val_table[idx_lower + 1]

                return val_lower + t * (val_upper - val_lower)
        else:
            if self.device is not None:
                import torch
                if not isinstance(separation, torch.Tensor):
                    separation = torch.tensor(separation, dtype=torch.float32, device=self.device)
                elif separation.device != torch.device(self.device):
                    separation = separation.to(self.device)

                x = torch.pi * separation / self.L0_t
                y = x ** (5 / 3)
                x1 = x ** 2

                series = torch.zeros_like(separation)
                xn = torch.ones_like(separation)

                for n in range(1, self.n_max + 1):
                    xn = xn * x1
                    temp = y * self.a[n] + self.b[n]
                    series = series + temp * xn

                temp = y * self.a[0] + series
                temp = temp * self.prefactor_t

                return temp
            else:
                return self._compute_exact_numpy(separation)