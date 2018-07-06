'''
Statistics tools for tracking.

MIT License

Copyright (c) 2018 Standard Cognition

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

# Standard
from sys import float_info  # for float_info.epsilon
from typing import Union, Tuple, List

# Scientific Computing
import numpy as np
from scipy.stats import chi2
from matplotlib import pyplot as plt
from matplotlib import patches


# ----- Multivariate Normal Distributions -----


def assert_cov_validity(
        cov: Union[float, np.ndarray],
        eigenvalue_lbnd: float = 1000.0*float_info.epsilon,
        condition_number_ubnd: float = 1.0e6):
    '''
    Assert that covariance `cov` is
        symmetric,
        real,
        positive-definite,
        has eigenvalues not too close to zero, and
        is well-conditioned.

    ::WARNING:: Applying `enforce_cov_validity` with the same parameters
    does not guarantee that these assertions will pass. Consider either (1)
    using the functions mutally exclusively, or (2) making the parameters of
    `enforce_cov_validity` slightly stricter in order to compensate for
    possible small numerical errors in eigenreconstruction.

    Args:
        cov: an alleged variance (as `float`) or covariance matrix (as
          `np.ndarray`).
        eigenvalue_lbnd: eigenvalues should be at least this much greater than
          zero. Must be strictly positive.
        condition_number_ubnd: inclusive upper bound on matrix condition
          number. Must be greater or equal to 1.0.

    Returns:
        Whether cov is positive definite and has all real elements.
    '''
    assert eigenvalue_lbnd > 0.0, \
        'Covariance eigenvalue lower bound must be > 0.0!'
    assert condition_number_ubnd >= 1.0, \
        'Covariance condition number bound must be >= 1.0!'

    # Symmetry
    if not np.isscalar(cov):
        assert (cov.T == cov).all(), 'Covariance must be symmetric!'

    # Realness
    assert np.isrealobj(cov), 'Covariance must be a real object!'

    # Eigenvalue properties
    if np.isscalar(cov):
        assert cov > 0.0, \
                'Variance must be strictly positive!'
        assert cov >= eigenvalue_lbnd, \
                'Variance must be >= lower bound!'
    else:
        # Precompute eigenvalues for subsequent tests.
        ws = np.linalg.eigvalsh(cov)  # The eigenvalues of cov
        w_min = min(ws)
        w_max = max(ws)

        # Strict positivity
        assert w_min > 0.0, 'Covariance must be strictly positive!'

        # Eigenvalue lower bound
        assert w_min >= eigenvalue_lbnd, \
            'Covariance eigenvalues must be >= lower bound!'

        # Condition number upper bound
        assert w_max/w_min <= condition_number_ubnd, \
            'Condition number must be <= upper bound!'


def enforce_cov_validity(
        cov: Union[float, np.ndarray],
        eigenvalue_lbnd: float = 1000.0*float_info.epsilon,
        condition_number_ubnd: float = 1.0e6) -> Union[float, np.ndarray]:
    '''
    Create and return a version of covariance `cov` which is modified to
    ensure it is
        symmetric,
        real,
        positive-definite,
        has eigenvalues not too close to zero, and
        is well-conditioned.

    ::WARNING:: Applying this function to a numpy array does not guarantee that
    calling `assert_cov_validity` with the same parameters will pass.
    Consider either (1) using the functions mutally exclusively, or (2) making
    the parameters of `assert_cov_validity` slightly more lenient in
    order to compensate for possible small numerical errors in
    eigenreconstruction.

    Args:
        cov: an alleged variance (as `float`) or covariance matrix (as
          `np.ndarray`).
        eigenvalue_lbnd: eigenvalues should be at least this much greater than
          zero.
        condition_number_ubnd: upper bound on matrix condition number. Should
          be greater or equal to 1.0. If it is necessary to modify `cov` to
          enforce this, the largest eigenvalue is held fixed and the smaller
          are increased.

    Returns:
        A version of cov modified to be valid.
    '''
    assert eigenvalue_lbnd > 0.0, \
        'Covariance eigenvalue lower bound must be > 0.0!'
    assert condition_number_ubnd >= 1.0, \
        'Covariance condition number bound must be >= 1.0!'

    if np.isscalar(cov):
        # Realness
        cov = float(cov.real)

        # Eigenvalue lower bound
        if cov < eigenvalue_lbnd:
            cov = eigenvalue_lbnd
    else:
        # Symmetry
        cov = 0.5*(cov + cov.T)

        # Realness
        if not np.isrealobj(cov):
            cov = cov.real

        # Precompute eigendecomposition for subsequent enforcements.
        ws, vr = np.linalg.eigh(cov)  # Eigenvalues and right eigenvectors

        # Eigenvalue lower bound
        for i, w in enumerate(ws):
            if w < eigenvalue_lbnd:
                ws[i] = eigenvalue_lbnd

        # Condition number upper bound
        # condition number := max_eigval/min_eigval <= condition_number_ubnd
        # <=> max_eigval/condition_number_ubnd <= min_eigval
        eigenvalue_lbnd_for_conditioning = max(ws)/condition_number_ubnd
        for i, w in enumerate(ws):
            if w < eigenvalue_lbnd_for_conditioning:
                ws[i] = eigenvalue_lbnd_for_conditioning

        # Eigenreconstruction
        cov = vr.dot(np.diag(ws).dot(vr.T))

    return cov


def evaluate_normal_pdf(
        x: Union[float, np.ndarray],
        cov: Union[float, np.ndarray],
        mean: Union[float, np.ndarray] = None) -> float:
    '''
    Compute and return the value of a multivariate normal PDF (Probability
    Density Function) at a point x.

    Args:
        x: where to evaluate PDF.
        cov: covariance of distribution.
        mean: mean of distribution. None => assumed zeros.

    Returns:
        PDF value at x.
    '''
    # Get dimension of distribution
    if np.isscalar(x):
        dimension = 1
    else:
        dimension = len(x)

    if mean is None:
        delta = x  # assume zero mean
    else:
        delta = x - mean

    if dimension > 1:
        k = (2.0*np.pi)**(-0.5*dimension)*np.linalg.det(cov)**(-0.5)
        quadratic = delta.dot(np.linalg.solve(cov, delta))
        p = k*np.exp(-0.5*quadratic)
    else:
        k = (2.0*np.pi*cov)**(-0.5)
        quadratic = delta*(1.0/cov)*delta
        p = k*np.exp(-0.5*quadratic)

    return float(p)


def sample_from_normal_distribution(
        cov: Union[float, np.ndarray],
        cov_cholesky: np.ndarray = None,
        mean: Union[float, np.ndarray] = None,
        num_samples: int = 1) -> np.ndarray:
    '''
    Generate random sample(s) from a normal distribution having mean `mean`
    and covariance `cov`.

    This function is used instead of `np.random.multivariate_normal` because
    the latter issues incorrect warnings (as of 2018:05:24) and is less
    flexible in input. It may also be less efficient if you already have a
    Cholesky factorization.

    Args:
        cov: covariance of the distribution.
        cov_cholesky: optionally precomputed cholesky factorization, as output
          from `np.linalg.cholesky(cov)`. If `cov_cholesky` is None, then the
          covariance is allowed to be rank deficient.
        mean: mean of the distribution. None => assume zeros.
        num_samples: number of desired samples.

    Returns:
        Array of samples. Each column is a sample and the rows run over
        components of the vectors.
    '''
    if np.isscalar(cov):
        sigma = np.sqrt(cov)
        samples = sigma*np.random.normal(size=(1, num_samples)) + mean
    else:
        d = cov.shape[0]
        if mean is None:
            mean = np.zeros(d)
        try:
            if cov_cholesky is None:
                cov_cholesky = np.linalg.cholesky(cov)
            samples = np.dot(
                cov_cholesky, np.random.normal(size=(d, num_samples)))
            for i in range(d):
                samples[i, :] += mean[i]
        except np.linalg.linalg.LinAlgError:
            # Fall back on `np.random.multivariate_normal` only for rank-
            # deficient covariances.
            samples = np.random.multivariate_normal(
                mean=mean, cov=cov, size=num_samples)
            samples = samples.T

    return samples




# ----- Error Ellipse Visualization -----


def generate_error_ellipse_points(
        mean: np.ndarray,
        cov: np.ndarray,
        cov_cholesky: np.ndarray = None,
        acceptance: float = 0.99,
        num_points: int = 30) -> np.ndarray:
    '''
    Generate points on a level set of a bivariate Gaussian PDF, usu. for
    plotting error ellipses.

    Args:
        mean: the distribution's mean.
        cov: 2x2 array, the distribution's covariance.
        cov_cholesky: optionally precomputed cholesky factorization, as output
          from `np.linalg.cholesky(cov)`.
        acceptance: probability mass that ellipse should contain around mean.
        num_points: number of points to sample on ellipse. This is a measure of
          plotting resolution.

    Returns:
        Shape (2, num_points) array of points for plotting.
    '''
    assert mean.shape == (2,), 'Incorrect mean shape!'
    assert cov.shape == (2, 2), 'Incorrect cov shape!'
    assert acceptance >= 0.0 and acceptance < 1.0, \
        'acceptance rate must be in [0.0, 1.0)!'

    # Sample points on unit circle.
    dtheta = 2.0*np.pi/num_points
    thetas = np.linspace(0, 2.0*np.pi - dtheta, num_points)
    if cov_cholesky is None:
        cov_cholesky = np.linalg.cholesky(cov)
    acceptance_factor = np.sqrt(chi2.ppf(acceptance, df=2))
    cov_cholesky = acceptance_factor*cov_cholesky
    points = np.zeros((2, num_points))
    points[0, :] = np.cos(thetas)
    points[1, :] = np.sin(thetas)

    # Warp circle points into ellipse.
    for i in range(num_points):
        points[:, i] = cov_cholesky.dot(points[:, i]) + mean

    return points


def plot_polygon(
        boundary_points: np.ndarray,
        edgecolor: str = 'k', facecolor: str = 'm', alpha: float = 0.5,
        linewidth: float = 3.0, linestyle: str = 'solid', zorder: int = 0):
    '''
    Wrapper for `plt.fill` that has reasonable default arguments for
    plotting acceptance regions, esp. error ellipses.

    Args:
        boundary_points: shape 2 x many, no repeat at wraparound. First row is
          x values, second is y values.
        zorder: higher => closer to foreground.
    '''
    plt.fill(
        boundary_points[0, :], boundary_points[1, :],
        edgecolor=edgecolor, facecolor=facecolor, alpha=alpha,
        linewidth=linewidth, linestyle=linestyle, zorder=zorder)


def plot_error_ellipse(
        mean: np.ndarray,
        cov: np.ndarray,
        cov_cholesky: np.ndarray = None,
        acceptance: float = 0.99,
        num_points: int = 30,
        edgecolor: str = 'k', facecolor: str = 'm', alpha: float = 0.5,
        linewidth: float = 3.0, linestyle: str = 'solid', zorder: int = 0) \
        -> List[patches.Polygon]:
    '''
    Plot 2D error ellipse from mean and covariance.

    Args:
        mean: distribution's mean (length 2).
        cov: distribution's covariance (2x2).
        acceptance: amount of probability mass ellipse should contain.
        edgecolor: edge color.
        facecolor: face color, 'none' => transparent interior.
        alpha: close to 0.0 => transparent, close to 1.0 => opaque.
        linewidth: usu. 3.0 or greater for good visibility.
        linestyle: e.g. '-', '--', or ':'.
        zorder: higher => closer to foreground.

    Returns:
        list of ...?
    '''
    if cov_cholesky is None:
        cov_cholesky = np.linalg.cholesky(cov)
    boundary_points = generate_error_ellipse_points(
        mean=mean, cov=cov, cov_cholesky=cov_cholesky,
        acceptance=acceptance, num_points=num_points)

    polygons = plt.fill(
        boundary_points[0, :], boundary_points[1, :],
        edgecolor=edgecolor, facecolor=facecolor, alpha=alpha,
        linewidth=linewidth, linestyle=linestyle, zorder=zorder)[0]

    return polygons
