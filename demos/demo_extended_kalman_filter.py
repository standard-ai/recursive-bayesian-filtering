#!/usr/bin/env python3
'''
2D EKF Demo.

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
# pylint: disable=W0611

# Standard.
import time

# Scientific Computing and Visualization.
import numpy as np
from matplotlib import pyplot as plt

# Custom.
import stats_tools as stm
import dynamic_models as dmm
import measurements as mm
import extended_kalman_filter as ekfm


def main():
    #np.random.seed(1)
    np.random.seed(int(time.time()))
    start_time = time.time()
    plt.rcParams['figure.figsize'] = 10, 10  # Set default fig size.

    # Discrete times.
    framerate = 100  # fps.
    dt = 1.0/framerate  # seconds between consecutive frames.
    num_frames = 10  # Includes initial frame.

    # Dynamic model.
    d = 4  # Planar NCV state space dimension.
    ncv = dmm.NcvContinuous(dimension=d, sa2=2.0)

    # Truth states.
    xs_truth = np.zeros((d, num_frames))
    theta0_truth = 0.0  # True initial heading.
    xs_truth[:, 0] = \
        np.array((0.0, 0.0, np.cos(theta0_truth), np.sin(theta0_truth)))
    for frame_num in range(1, num_frames):
        dx = ncv.sample_process_noise(dt)
        xs_truth[:, frame_num] = ncv(xs_truth[:, frame_num-1], dt=dt) + dx

    # Measurements.
    # Max error 0.1 m => use 0.05 as standard deviation => 0.0025 variance.
    measurements = []
    R = 0.00001*np.eye(d//2)
    R_cholesky = np.linalg.cholesky(R)
    dzs = stm.sample_from_normal_distribution(
        cov=R, cov_cholesky=R_cholesky, num_samples=num_frames)
    frame_num = 0
    for x, dz in zip(xs_truth.T, dzs.T):
        z = x[:d//2] + dz
        measurements.append(
            mm.PositionMeasurement(mean=z, cov=R, frame_num=frame_num))
        frame_num += 1

    # State estimates.
    x0 = np.zeros(d)
    P0 = 100.0*np.eye(d)
    ekf_state_last = ekfm.EKFState(
        dynamic_model=ncv, mean=x0, cov=P0, frame_num=0)
    ekf_states = []
    for measurement in measurements:
        ekf_state = ekf_state_last.copy()
        ekf_state.predict(dt=dt, destination_frame_num=measurement.frame_num)
        dz, S = ekf_state.update(measurement=measurement)
        ekf_states.append(ekf_state)
        ekf_state_last = ekf_state

    # Plot truth.
    plt.plot(
        xs_truth[0, :], xs_truth[1, :], 'g.-',
        markersize=15, linewidth=2.0, label='truth', zorder=2)

    # Plot measurements and tracks.
    acceptance = 0.99
    #
    # Measurements.
    plt.plot(
        [ measurement.mean[0] for measurement in measurements],
        [ measurement.mean[1] for measurement in measurements],
        'o-', color='#00bfff', markersize=5, markerfacecolor='y',
        markeredgewidth=1.0, markeredgecolor='k', label='measurement',
        zorder=4)
    for measurement in measurements:
        stm.plot_error_ellipse(
            mean=measurement.mean, cov=measurement.cov,
            cov_cholesky=R_cholesky,
            acceptance=acceptance,
            num_points=30, edgecolor='k', facecolor='m', alpha=0.5,
            linewidth=2.0, linestyle='solid', zorder=0)
    #
    # Tracks.
    plt.plot(
        [ekf_state.mean[0] for ekf_state in ekf_states],
        [ekf_state.mean[1] for ekf_state in ekf_states], 'o-',
        color='#ffb818', mec='w', mfc='k', ms=8, lw=2.0, label='track',
        zorder=3)
    for ekf_state in ekf_states:
        stm.plot_error_ellipse(
            mean=ekf_state.mean[:2], cov=ekf_state.cov[:2, :2],
            cov_cholesky=None, acceptance=acceptance,
            num_points=30, edgecolor='k', facecolor='k', alpha=0.6,
            linewidth=2.0, linestyle='solid', zorder=1)

    # Plot aesthetics.
    ax = plt.gca()
    ax.set_aspect('equal')
    plt.title('EKF Demo', fontsize=16, fontweight='bold')
    plt.xlabel('x', fontsize=14, fontweight='bold')
    plt.ylabel('y', fontsize=14, fontweight='bold')
    legend = plt.legend(numpoints=1)
    plt.setp(legend.get_texts(), fontsize='14', fontweight='bold')
    ax.margins(0.2)  # Only works if limits not manually set.
    plt.grid(True)
    plt.show()

    print('Elapsed time =', time.time() - start_time)


if __name__ == '__main__':
    main()
