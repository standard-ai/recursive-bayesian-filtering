'''
Test `stats_tools`.

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

# pylint: disable=C0413, R0915

# Standard
import os
from sys import float_info  # for float_info.epsilon
import pytest

# Scientific Computing
import numpy as np
from scipy.stats import chi2
from scipy.stats import entropy
from scipy.stats import norm
from matplotlib import pyplot as plt
plt.rcParams['figure.figsize'] = 7, 7  # Default fig size.

# Custom
import stats_tools as stm


def test_assert_and_enforce_cov_validity():
    # Correct scalar
    P = 1.0
    stm.assert_cov_validity(P)

    # Complex scalar
    P = 1.0 + 1.0j
    with pytest.raises(AssertionError):
        stm.assert_cov_validity(P)
    Q = stm.enforce_cov_validity(P)
    stm.assert_cov_validity(
        Q, eigenvalue_lbnd=900.0*float_info.epsilon,
        condition_number_ubnd=1.0e7)
    with pytest.raises(AssertionError):
        # Make sure P wasn't corrected in-place.
        stm.assert_cov_validity(P)

    # Not positive definite scalar
    P = -1.0
    with pytest.raises(AssertionError):
        stm.assert_cov_validity(P)
    Q = stm.enforce_cov_validity(P)
    stm.assert_cov_validity(
        Q, eigenvalue_lbnd=900.0*float_info.epsilon,
        condition_number_ubnd=1.0e7)
    with pytest.raises(AssertionError):
        # Make sure P wasn't corrected in-place.
        stm.assert_cov_validity(P)

    # Not psitive enough scalar
    P = 500.0*float_info.epsilon
    with pytest.raises(AssertionError):
        stm.assert_cov_validity(P)
    Q = stm.enforce_cov_validity(P)
    stm.assert_cov_validity(
        Q, eigenvalue_lbnd=900.0*float_info.epsilon,
        condition_number_ubnd=1.0e7)
    with pytest.raises(AssertionError):
        # Make sure P wasn't corrected in-place.
        stm.assert_cov_validity(P)

    # Correct matrix
    P = np.array([[1.0, 0.0], [0.0, 1.0]])
    stm.assert_cov_validity(P)

    # Complex matrix
    P = np.array([[1.0 + 2.0j, 0.0], [0.0, 1.0]])
    with pytest.raises(AssertionError):
        stm.assert_cov_validity(P)
    Q = stm.enforce_cov_validity(P)
    stm.assert_cov_validity(
        Q, eigenvalue_lbnd=900.0*float_info.epsilon,
        condition_number_ubnd=1.0e7)
    with pytest.raises(AssertionError):
        # Make sure P wasn't corrected in-place.
        stm.assert_cov_validity(P)

    # Asymmetric matrix
    P = np.array([[1.0, 0.1], [0.0, 1.0]])
    with pytest.raises(AssertionError):
        stm.assert_cov_validity(P)
    Q = stm.enforce_cov_validity(P)
    stm.assert_cov_validity(
        Q, eigenvalue_lbnd=900.0*float_info.epsilon,
        condition_number_ubnd=1.0e7)
    with pytest.raises(AssertionError):
        # Make sure P wasn't corrected in-place.
        stm.assert_cov_validity(P)

    # Not positive definite matrix
    P = np.array([[-1.0, 0.0], [0.0, -1.0]])
    with pytest.raises(AssertionError):
        stm.assert_cov_validity(P)
    Q = stm.enforce_cov_validity(P)
    stm.assert_cov_validity(
        Q, eigenvalue_lbnd=900.0*float_info.epsilon,
        condition_number_ubnd=1.0e7)
    with pytest.raises(AssertionError):
        # Make sure P wasn't corrected in-place.
        stm.assert_cov_validity(P)

    # Not positive enough matrix
    P = np.array(
        [[500.0*float_info.epsilon, 0.0], [0.0, 500.0*float_info.epsilon]])
    with pytest.raises(AssertionError):
        stm.assert_cov_validity(P)
    Q = stm.enforce_cov_validity(P)
    stm.assert_cov_validity(
        Q, eigenvalue_lbnd=900.0*float_info.epsilon,
        condition_number_ubnd=1.0e7)
    with pytest.raises(AssertionError):
        # Make sure P wasn't corrected in-place.
        stm.assert_cov_validity(P)

    # Bad condition number matrix
    P = np.array([[1.0, 0.0], [0.0, 10000.0*float_info.epsilon]])
    with pytest.raises(AssertionError):
        stm.assert_cov_validity(P)
    Q = stm.enforce_cov_validity(P)
    stm.assert_cov_validity(
        Q, eigenvalue_lbnd=900.0*float_info.epsilon,
        condition_number_ubnd=1.0e7)
    with pytest.raises(AssertionError):
        # Make sure P wasn't corrected in-place.
        stm.assert_cov_validity(P)


def test_evaluate_normal_pdf():
    # Univariate
    x = 0.1
    mean = 0.0
    cov = 1.0
    p_ref = norm.pdf(x, loc=mean, scale=np.sqrt(cov))
    p = stm.evaluate_normal_pdf(x, cov, mean)
    assert np.isclose(p, p_ref)
    p = stm.evaluate_normal_pdf(x, cov)
    assert np.isclose(p, p_ref)

    # Multivariate
    x = np.array([1.0, 0.0])
    mean = np.array([0.0, 0.0])
    cov = np.array([
        [1.0, 0.0],
        [0.0, 1.0]])
    p_ref = 1./(2*np.pi)*np.exp(-0.5)  # From analytical PDF formula
    p = stm.evaluate_normal_pdf(x, cov, mean)
    assert np.isclose(p, p_ref)
    p = stm.evaluate_normal_pdf(x, cov)
    assert np.isclose(p, p_ref)


def test_sample_from_normal_distribution():
    num_samples = 10000

    # Univariate
    cov = 2.0
    mean = 1.5
    samples = stm.sample_from_normal_distribution(
        cov=cov, mean=mean, num_samples=num_samples)
    # Check against mean of Chi^2 distribution with 1 DoF.
    chi2_samples = (samples - mean)**2/cov
    assert abs(np.mean(chi2_samples) - 1.0) < 0.05, \
        'MC sampling did not confirm univariate normal statistics!'

    # Full-Rank Multivariate
    cov = np.array([
        [10.0, 0.2],
        [0.1, 5.0]])
    mean = np.array([2.4, 8.8])
    cov_cholesky = np.linalg.cholesky(cov)
    samples = stm.sample_from_normal_distribution(
        cov=cov, cov_cholesky=cov_cholesky, mean=mean, num_samples=num_samples)
    # Check against mean of Chi^2 distribution with 2 DoF.
    chi2_samples = np.zeros(num_samples)
    for i in range(num_samples):
        sample = samples[:, i]
        chi2_samples[i] = (sample - mean).dot(np.linalg.solve(cov, sample - mean))
    assert abs(np.mean(chi2_samples) - 2.0) < 0.1, \
        'MC sampling did not confirm multivariate normal statistics!'

    # Rank-Deficient Multivariate
    cov = np.array([
        [1.0, 0.0],
        [0.0, 0.0]])
    mean = np.array([2.4, 8.8])
    samples = stm.sample_from_normal_distribution(
        cov=cov, mean=mean, num_samples=num_samples)
    assert samples.shape == (2, num_samples)
    chi2_samples = np.zeros(num_samples)
    for i in range(num_samples):
        sample = samples[0, i]
        chi2_samples[i] = (sample - mean[0])*(sample - mean[0])
    assert abs(np.mean(chi2_samples) - 1.0) < 0.1, \
        'MC sampling did not confirm multivariate normal statistics!'


def test_plot_error_ellipse():
    '''
    Indirectly tests also `generate_error_ellipse_points` and
    `plot_polygon`.

    Cf. `../demos/demo_error_ellipse_plot.py`.
    '''
    plt.clf()

    acceptance = 0.8
    boundary_num_points = 50
    scatter_num_points = 2000

    # Mean
    mean = np.array([2.4, 8.8])

    # Covariance
    r1 = 1.0
    r2 = 3.0
    cov = np.array([[1.0/r1**2, 0.0], [0.0, 1.0/r2**2]])
    theta = 0.8 * np.pi
    R = np.array(
        [[np.cos(theta), -np.sin(theta)],
         [np.sin(theta), np.cos(theta)]])
    cov = np.dot(np.dot(R, cov), R.T)
    cov_cholesky = np.linalg.cholesky(cov)

    # Initialize plot
    ax = plt.gca()
    ax.set_aspect('equal')

    # Plot error ellipse.
    stm.plot_error_ellipse(
        mean, cov, cov_cholesky, acceptance, boundary_num_points,
        edgecolor='k', facecolor='m', alpha=0.5,
        linewidth=3.0, linestyle='solid', zorder=0)
