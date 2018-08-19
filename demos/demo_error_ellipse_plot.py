#!/usr/bin/env python3
'''Demo of error ellipse plotting using `stats_tools`.'''
# pylint: disable=W0611

# Standard
import time

# Scientific
import numpy as np
from scipy.stats import chi2
import matplotlib
matplotlib.use('tkagg')  # For plotting to work over X forwarding to a Mac.
from matplotlib import pyplot as plt

# Custom
import stats_tools as stm


def main():
    np.random.seed(1)
    start_time = time.time()
    plt.rcParams['figure.figsize'] = 10, 10  # Set default fig size.
    plt.clf()

    acceptance = 0.8
    boundary_num_points = 50
    num_samples = 2000

    # Mean
    mean = np.array([2.4, 8.8])

    # Covariance
    r1 = 1.0
    r2 = 3.0
    cov = np.array([[1.0/r1**2, 0.0], [0.0, 1.0/r2**2]])
    theta = 0.8*np.pi
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

    # Verify ellipse with scatter samples.
    samples = stm.sample_from_normal_distribution(
        mean=mean, cov=cov,
        cov_cholesky=cov_cholesky, num_samples=num_samples)
    #
    # Color samples in and outside of ellipse differently.
    x_passed_samples = []
    y_passed_samples = []
    x_failed_samples = []
    y_failed_samples = []
    chi2_threshold = chi2.ppf(acceptance, df=2)
    for sample in samples.T:
        # Check quadratic form against acceptance.
        z = sample - mean
        chi2_stat = np.dot(z.T, np.linalg.solve(cov, z))
        if chi2_stat <= chi2_threshold:
            x_passed_samples.append(sample[0])
            y_passed_samples.append(sample[1])
        else:
            x_failed_samples.append(sample[0])
            y_failed_samples.append(sample[1])
    plt.scatter(
        x_passed_samples, y_passed_samples,
        marker='.', color='g', s=5.4, zorder=10)
    plt.scatter(
        x_failed_samples, y_failed_samples,
        marker='.', color='r', s=5.4, zorder=10)

    # Plot Formatting
    r_plot = 4.0
    plt.xlim((mean[0]-r_plot, mean[0]+r_plot))
    plt.ylim((mean[1]-r_plot, mean[1]+r_plot))
    plt.grid(True)
    plt.title(
        'Error Ellipse Demo',
        fontsize=16, fontweight='bold')
    plt.xlabel('x', fontsize=14, fontweight='bold')
    plt.ylabel('y', fontsize=14, fontweight='bold')
    ax.margins(0.1)
    plt.grid(True)
    plt.show()

    print('Elapsed time =', time.time() - start_time)


if __name__ == '__main__':
    main()
