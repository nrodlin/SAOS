import numpy as np
from scipy.special import gamma


class StructureFunctionVK:
    """
    Von Karman phase structure function.

    This implementation follows the finite outer-scale expansion used in
    pseudo-analytical AO covariance calculations.
    """

    def __init__(self, r0: float, L0: float, n_max: int = 10, device: str | None = None):
        self.r0 = r0
        self.L0 = L0
        self.n_max = n_max
        self.device = device

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

        if self.device is not None:
            import torch
            self.a = torch.tensor(a_np, dtype=torch.float32, device=self.device)
            self.b = torch.tensor(b_np, dtype=torch.float32, device=self.device)
            self.prefactor_t = torch.tensor(self.prefactor, dtype=torch.float32, device=self.device)
            self.L0_t = torch.tensor(self.L0, dtype=torch.float32, device=self.device)
        else:
            self.a = a_np
            self.b = b_np

    def __call__(self, separation):
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