"""Geometry module - Riemannian geometry computations for IGQK."""

from .fisher import FisherMetric
from .geodesic import GeodesicSolver
from .laplacian import LaplaceBeltrami

__all__ = ["FisherMetric", "GeodesicSolver", "LaplaceBeltrami"]
