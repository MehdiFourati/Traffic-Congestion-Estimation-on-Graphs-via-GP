from __future__ import annotations

import torch
import gpytorch


class GraphMaternKernel(gpytorch.kernels.Kernel):
    """
    Graph Matérn-style kernel based on a graph Laplacian eigendecomposition.

    Inputs
    ------
    x1, x2 : torch.Tensor
        Expected shape (..., n, 1) or (..., n), containing node indices.

    Parameters
    ----------
    eigenvalues : torch.Tensor
        Shape (N,), graph Laplacian eigenvalues.
    eigenvectors : torch.Tensor
        Shape (N, N), graph Laplacian eigenvectors.

    Trainable hyperparameters
    -------------------------
    nu : smoothness-like exponent in the spectral filter
    kappa : lengthscale-like parameter in the spectral filter

    Spectral form
    -------------
        w_k = (lambda_k + c / kappa^2)^(-nu),  c = 2 * nu

        K(i, j) = sum_k w_k * phi_k(i) * phi_k(j)
    """

    has_lengthscale = False

    def __init__(
        self,
        eigenvalues: torch.Tensor,
        eigenvectors: torch.Tensor,
        eps: float = 1e-8,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if eigenvalues.ndim != 1:
            raise ValueError(f"eigenvalues must be 1D, got shape {eigenvalues.shape}")
        if eigenvectors.ndim != 2:
            raise ValueError(f"eigenvectors must be 2D, got shape {eigenvectors.shape}")
        if eigenvectors.shape[0] != eigenvectors.shape[1]:
            raise ValueError(f"eigenvectors must be square, got shape {eigenvectors.shape}")
        if eigenvectors.shape[0] != eigenvalues.shape[0]:
            raise ValueError(
                f"Mismatch: eigenvectors has {eigenvectors.shape[0]} rows but "
                f"eigenvalues has length {eigenvalues.shape[0]}"
            )

        self.eps = float(eps)

        self.register_buffer("eigenvalues", eigenvalues.clone().detach().float())
        self.register_buffer("eigenvectors", eigenvectors.clone().detach().float())

        self.register_parameter(
            name="raw_nu",
            parameter=torch.nn.Parameter(torch.tensor(0.5))
        )
        self.register_parameter(
            name="raw_kappa",
            parameter=torch.nn.Parameter(torch.tensor(1.0))
        )

        positive_constraint = gpytorch.constraints.Positive()
        self.register_constraint("raw_nu", positive_constraint)
        self.register_constraint("raw_kappa", positive_constraint)

    @property
    def nu(self) -> torch.Tensor:
        return self.raw_nu_constraint.transform(self.raw_nu)

    @nu.setter
    def nu(self, value: float | torch.Tensor) -> None:
        if not torch.is_tensor(value):
            value = torch.tensor(value, dtype=self.raw_nu.dtype, device=self.raw_nu.device)
        self.initialize(raw_nu=self.raw_nu_constraint.inverse_transform(value))

    @property
    def kappa(self) -> torch.Tensor:
        return self.raw_kappa_constraint.transform(self.raw_kappa)

    @kappa.setter
    def kappa(self, value: float | torch.Tensor) -> None:
        if not torch.is_tensor(value):
            value = torch.tensor(value, dtype=self.raw_kappa.dtype, device=self.raw_kappa.device)
        self.initialize(raw_kappa=self.raw_kappa_constraint.inverse_transform(value))

    def spectral_weights(self) -> torch.Tensor:
        nu = self.nu
        kappa = self.kappa

        c = 2.0 * nu
        denom = self.eigenvalues + c / (kappa ** 2 + self.eps)
        weights = torch.pow(denom.clamp_min(self.eps), -nu)
        return weights

    def _extract_node_indices(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim >= 2 and x.shape[-1] == 1:
            x = x.squeeze(-1)
        return x.long()

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        diag: bool = False,
        **params,
    ) -> torch.Tensor:
        idx1 = self._extract_node_indices(x1)
        idx2 = self._extract_node_indices(x2)

        phi1 = self.eigenvectors[idx1]   # (..., n1, N)
        phi2 = self.eigenvectors[idx2]   # (..., n2, N)

        w = self.spectral_weights()      # (N,)
        phi1_w = phi1 * w

        K = phi1_w @ phi2.transpose(-1, -2)

        if diag:
            return torch.diagonal(K, dim1=-2, dim2=-1)

        return K


class GraphTemporalKernel(gpytorch.kernels.Kernel):
    """
    Spatio-temporal kernel with input format:

        x[..., 0] = node index
        x[..., 1] = time

    Kernel:
        K((i,t), (j,s)) = K_graph(i,j) * K_time(t,s)

    Notes
    -----
    - The graph part is handled by GraphMaternKernel.
    - The time part is any standard GPyTorch kernel, e.g. MaternKernel or RBFKernel.
    """

    def __init__(
        self,
        eigenvalues: torch.Tensor,
        eigenvectors: torch.Tensor,
        time_kernel: gpytorch.kernels.Kernel | None = None,
        eps: float = 1e-8,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.graph_kernel = GraphMaternKernel(
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            eps=eps,
        )

        self.time_kernel = (
            time_kernel if time_kernel is not None
            else gpytorch.kernels.MaternKernel(nu=1.5)
        )

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        diag: bool = False,
        **params,
    ) -> torch.Tensor:
        if x1.shape[-1] != 2 or x2.shape[-1] != 2:
            raise ValueError(
                f"Expected inputs with last dimension 2: [node_idx, time]. "
                f"Got {x1.shape} and {x2.shape}"
            )

        node1 = x1[..., 0].unsqueeze(-1)   # (..., n1, 1)
        node2 = x2[..., 0].unsqueeze(-1)   # (..., n2, 1)

        time1 = x1[..., 1].unsqueeze(-1)   # (..., n1, 1)
        time2 = x2[..., 1].unsqueeze(-1)   # (..., n2, 1)

        Kg = self.graph_kernel(node1, node2, diag=diag, **params)
        Kt = self.time_kernel(time1, time2, diag=diag, **params)

        return Kg * Kt