'''
Test `extended_kalman_filter`.

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

# pylint: disable=C0413, W0611

# Standard
import pytest

# Scientific Computing
import numpy as np
from matplotlib import pyplot as plt
plt.rcParams['figure.figsize'] = 7, 7  # Default fig size.

# Custom
#import stats_tools as stm
import measurements as mm
import dynamic_models as dmm
import extended_kalman_filter as ekfm


def test_EKFState_with_NcpContinuous():
    d = 3
    ncp = dmm.NcpContinuous(dimension=d, sv2=2.0)
    x = np.random.random(d)
    P = np.eye(d)
    t = 0.0
    dt = 2.0
    ekf_state = ekfm.EKFState(dynamic_model=ncp, mean=x, cov=P, time=t)

    assert ekf_state.dynamic_model.__class__ == dmm.NcpContinuous
    assert ekf_state.dimension == d
    assert ekf_state.dimension_pv == 2*d

    assert np.allclose(x, ekf_state.mean)
    assert np.allclose(P, ekf_state.cov)
    assert np.allclose(x, ekf_state.mean_pv[:d])
    assert np.allclose(P, ekf_state.cov_pv[:d, :d])
    assert np.allclose(t, ekf_state.time)

    ekf_state.init(2*x, 2*P, t + 2.0)
    assert np.allclose(2*x, ekf_state.mean)
    assert np.allclose(2*P, ekf_state.cov)
    assert np.allclose(t + 2.0, ekf_state.time)

    ekf_state.init(2*x, 2*P, t)
    ekf_state1 = ekf_state.copy()
    ekf_state1.predict(dt)
    assert ekf_state1.dynamic_model.__class__ == dmm.NcpContinuous

    measurement = mm.PositionMeasurement(
        mean=np.random.random(d),
        cov=np.eye(d),
        time=t + dt)
    chi2_stat = ekf_state1.chi2_stat_of_update(measurement)
    assert chi2_stat >= 0.0
    l = ekf_state1.likelihood_of_update(measurement)
    assert l >= 0.0 and l <= 1.0
    old_mean = ekf_state1.mean.copy()
    dz, S = ekf_state1.update(measurement)
    assert dz.shape == (measurement.dimension,)
    assert S.shape == (measurement.dimension, measurement.dimension)
    assert not np.allclose(ekf_state1.mean, old_mean)

    ekf_state2 = ekf_state1.copy()
    assert ekf_state2.dynamic_model.__class__ == dmm.NcpContinuous


def test_EKFState_with_NcvContinuous():
    d = 6
    ncv = dmm.NcvContinuous(dimension=d, sa2=2.0)
    x = np.random.random(d)
    P = np.eye(d)
    t = 0.0
    dt = 2.0
    ekf_state = ekfm.EKFState(
        dynamic_model=ncv, mean=x, cov=P, time=t)

    assert ekf_state.dynamic_model.__class__ == dmm.NcvContinuous
    assert ekf_state.dimension == d
    assert ekf_state.dimension_pv == d

    assert np.allclose(x, ekf_state.mean)
    assert np.allclose(P, ekf_state.cov)
    assert np.allclose(x, ekf_state.mean_pv)
    assert np.allclose(P, ekf_state.cov_pv)
    assert np.allclose(t, ekf_state.time)

    ekf_state.init(2*x, 2*P, t + 2.0)
    assert np.allclose(2*x, ekf_state.mean)
    assert np.allclose(2*P, ekf_state.cov)
    assert np.allclose(t + 2.0, ekf_state.time)

    ekf_state.init(2*x, 2*P, t)
    ekf_state1 = ekf_state.copy()
    ekf_state1.predict(dt)
    assert ekf_state1.dynamic_model.__class__ == dmm.NcvContinuous

    measurement = mm.PositionMeasurement(
        mean=np.random.random(d),
        cov=np.eye(d),
        time=t + dt)
    chi2_stat = ekf_state1.chi2_stat_of_update(measurement)
    assert chi2_stat >= 0.0
    l = ekf_state1.likelihood_of_update(measurement)
    assert l >= 0.0 and l <= 1.0
    old_mean = ekf_state1.mean.copy()
    dz, S = ekf_state1.update(measurement)
    assert dz.shape == (measurement.dimension,)
    assert S.shape == (measurement.dimension, measurement.dimension)
    assert not np.allclose(ekf_state1.mean, old_mean)

    ekf_state2 = ekf_state1.copy()
    assert ekf_state2.dynamic_model.__class__ == dmm.NcvContinuous
