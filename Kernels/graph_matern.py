from __future__ import annotations

from typing import Optional

import torch
import gpytorch


class GraphMaternKernel(gpytorch.kernels.Kernel):
    """
    Inputs
    ------
    x1, x2 : torch.Tensor
        Expected shape (..., n, 1) or (..., n)
        
    Parameters
    ----------
    eigenvalues : torch.Tensor
        Shape (N,), graph Laplacian eigenvalues.
    eigenvectors : torch.Tensor
        Shape (N, N), graph Laplacian eigenvectors.
        Generated prior 

    Trainable hyperparameters
    -------------------------
    nu : smoothness-like exponent in the spectral filter
    kappa : lengthscale-like parameter in the spectral filter
    outputscale : overall kernel scale

    Spectral form
    -------------
    We use a graph-Matérn-style spectral weighting of the form

        w_k = (lambda_k + c / kappa^2)^(-nu)

    with c = 2 * nu

    and then

        K(i, j) = outputscale * sum_k w_k * phi_k(i) * phi_k(j)

    where phi_k are graph Laplacian eigenvectors.
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

        # Store spectral objects as buffers so they move with .to(device)
        self.register_buffer("eigenvalues", eigenvalues.clone().detach().float())
        self.register_buffer("eigenvectors", eigenvectors.clone().detach().float())

        # Raw trainable parameters
        self.register_parameter(
            name="raw_nu",
            parameter=torch.nn.Parameter(torch.tensor(0.5))
        )
        self.register_parameter(
            name="raw_kappa",
            parameter=torch.nn.Parameter(torch.tensor(1.0))
        )
        self.register_parameter(
            name="raw_outputscale",
            parameter=torch.nn.Parameter(torch.tensor(1.0))
        )

        positive_constraint = gpytorch.constraints.Positive()

        self.register_constraint("raw_nu", positive_constraint)
        self.register_constraint("raw_kappa", positive_constraint)
        self.register_constraint("raw_outputscale", positive_constraint)

    @property
    def nu(self) -> torch.Tensor:
        return self.raw_nu_constraint.transform(self.raw_nu)

    @nu.setter
    def nu(self, value: float | torch.Tensor) -> None:
        self._set_nu(value)

    def _set_nu(self, value: float | torch.Tensor) -> None:
        if not torch.is_tensor(value):
            value = torch.tensor(value, dtype=self.raw_nu.dtype, device=self.raw_nu.device)
        self.initialize(raw_nu=self.raw_nu_constraint.inverse_transform(value))

    @property
    def kappa(self) -> torch.Tensor:
        return self.raw_kappa_constraint.transform(self.raw_kappa)

    @kappa.setter
    def kappa(self, value: float | torch.Tensor) -> None:
        self._set_kappa(value)

    def _set_kappa(self, value: float | torch.Tensor) -> None:
        if not torch.is_tensor(value):
            value = torch.tensor(value, dtype=self.raw_kappa.dtype, device=self.raw_kappa.device)
        self.initialize(raw_kappa=self.raw_kappa_constraint.inverse_transform(value))

    @property
    def outputscale(self) -> torch.Tensor:
        return self.raw_outputscale_constraint.transform(self.raw_outputscale)

    @outputscale.setter
    def outputscale(self, value: float | torch.Tensor) -> None:
        self._set_outputscale(value)

    def _set_outputscale(self, value: float | torch.Tensor) -> None:
        if not torch.is_tensor(value):
            value = torch.tensor(
                value, dtype=self.raw_outputscale.dtype, device=self.raw_outputscale.device
            )
        self.initialize(
            raw_outputscale=self.raw_outputscale_constraint.inverse_transform(value)
        )

    def spectral_weights(self) -> torch.Tensor:
        """
        Compute spectral weights w_k for each graph eigenvalue lambda_k.
        """
        nu = self.nu
        kappa = self.kappa

        c = 2.0 * nu
        denom = self.eigenvalues + c / (kappa ** 2 + self.eps)
        weights = torch.pow(denom.clamp_min(self.eps), -nu)

        return self.outputscale * weights

    def _extract_node_indices(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert input tensor to a flat LongTensor of node indices.
        Accepts shape (n,), (n,1), or batched equivalents ending in 1.
        """
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
        """
        Compute K(x1, x2) where x1 and x2 contain node indices.
        """
        idx1 = self._extract_node_indices(x1)
        idx2 = self._extract_node_indices(x2)

        # Phi[idx] gives the selected rows of the eigenvector matrix
        phi1 = self.eigenvectors[idx1]   # shape (..., n1, N) or (n1, N)
        phi2 = self.eigenvectors[idx2]   # shape (..., n2, N) or (n2, N)

        w = self.spectral_weights()      # shape (N,)

        # Weight spectral features
        phi1_w = phi1 * w

        # K = Phi1 diag(w) Phi2^T
        K = phi1_w @ phi2.transpose(-1, -2)

        if diag:
            return torch.diagonal(K, dim1=-2, dim2=-1)

        return K