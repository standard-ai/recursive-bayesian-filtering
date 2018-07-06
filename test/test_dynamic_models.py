'''
Test `dynamic_models`.

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
import pytest

# Scientific Computing
import numpy as np

# Custom
import stats_tools as stm
import dynamic_models as dmm


def test_NcpContinuous():
    framerate = 100  # Hz
    dt = 1.0/framerate
    d = 3
    ncp = dmm.NcpContinuous(dimension=d, sv2=2.0)
    assert ncp.dimension == d
    assert ncp.dimension_pv == 2*d
    assert ncp.num_process_noise_parameters == 1

    x = np.random.random(d)
    y = ncp(x, dt)
    assert np.allclose(y, x)

    dx = ncp.geodesic_difference(x, y)
    assert np.allclose(dx, np.zeros(d))

    x_pv = ncp.mean2pv(x)
    assert len(x_pv) == 6
    assert np.allclose(x, x_pv[:d])
    assert np.allclose(np.zeros(d), x_pv[d:])

    P = np.eye(d)
    P_pv = ncp.cov2pv(P)
    assert P_pv.shape == (2*d, 2*d)
    P_pv_ref = np.zeros((2*d, 2*d))
    P_pv_ref[:d, :d] = P
    assert np.allclose(P_pv_ref, P_pv)

    Q = ncp.process_noise_cov(dt)
    Q1 = ncp.process_noise_cov(dt)  # Test caching.
    assert np.allclose(Q, Q1)
    assert Q1.shape == (d, d)
    stm.assert_cov_validity(Q1)

    dx = ncp.sample_process_noise(dt)
    assert dx.shape == (ncp.dimension,)

    ncp1 = ncp.copy()
    assert ncp1


def test_NcvContinuous():
    framerate = 100  # Hz
    dt = 1.0/framerate
    d = 6
    ncv = dmm.NcvContinuous(dimension=d, sa2=2.0)
    assert ncv.dimension == d
    assert ncv.dimension_pv == d
    assert ncv.num_process_noise_parameters == 1

    x = np.random.random(d)
    y = ncv(x, dt)
    assert np.allclose(y[0], x[0] + dt*x[d//2])

    dx = ncv.geodesic_difference(x, y)
    assert not np.allclose(dx, np.zeros(d))

    x_pv = ncv.mean2pv(x)
    assert len(x_pv) == d
    assert np.allclose(x, x_pv)

    P = np.eye(d)
    P_pv = ncv.cov2pv(P)
    assert P_pv.shape == (d, d)
    assert np.allclose(P, P_pv)

    Q = ncv.process_noise_cov(dt)
    Q1 = ncv.process_noise_cov(dt)  # Test caching.
    assert np.allclose(Q, Q1)
    assert Q1.shape == (d, d)
    stm.assert_cov_validity(Q1)

    dx = ncv.sample_process_noise(dt)
    assert dx.shape == (ncv.dimension,)

    ncv1 = ncv.copy()
    assert ncv1


def test_NcpDiscrete():
    framerate = 100  # Hz
    dt = 1.0/framerate
    d = 3
    ncp = dmm.NcpDiscrete(dimension=d, sv2=2.0)
    assert ncp.dimension == d
    assert ncp.dimension_pv == 2*d
    assert ncp.num_process_noise_parameters == 1

    x = np.random.random(d)
    y = ncp(x, dt)
    assert np.allclose(y, x)

    dx = ncp.geodesic_difference(x, y)
    assert np.allclose(dx, np.zeros(d))

    x_pv = ncp.mean2pv(x)
    assert len(x_pv) == 6
    assert np.allclose(x, x_pv[:d])
    assert np.allclose(np.zeros(d), x_pv[d:])

    P = np.eye(d)
    P_pv = ncp.cov2pv(P)
    assert P_pv.shape == (2*d, 2*d)
    P_pv_ref = np.zeros((2*d, 2*d))
    P_pv_ref[:d, :d] = P
    assert np.allclose(P_pv_ref, P_pv)

    Q = ncp.process_noise_cov(dt)
    Q1 = ncp.process_noise_cov(dt)  # Test caching.
    assert np.allclose(Q, Q1)
    assert Q1.shape == (d, d)
    stm.assert_cov_validity(Q1)

    dx = ncp.sample_process_noise(dt)
    assert dx.shape == (ncp.dimension,)

    ncp1 = ncp.copy()
    assert ncp1


def test_NcvDiscrete():
    framerate = 100  # Hz
    dt = 1.0/framerate
    dt = 100
    d = 6
    ncv = dmm.NcvDiscrete(dimension=d, sa2=2.0)
    assert ncv.dimension == d
    assert ncv.dimension_pv == d
    assert ncv.num_process_noise_parameters == 1

    x = np.random.random(d)
    y = ncv(x, dt)
    assert np.allclose(y[0], x[0] + dt*x[d//2])

    dx = ncv.geodesic_difference(x, y)
    assert not np.allclose(dx, np.zeros(d))

    x_pv = ncv.mean2pv(x)
    assert len(x_pv) == d
    assert np.allclose(x, x_pv)

    P = np.eye(d)
    P_pv = ncv.cov2pv(P)
    assert P_pv.shape == (d, d)
    assert np.allclose(P, P_pv)

    Q = ncv.process_noise_cov(dt)
    Q1 = ncv.process_noise_cov(dt)  # Test caching.
    assert np.allclose(Q, Q1)
    assert Q1.shape == (d, d)

    # Q has rank `dimension/2`, so it will not pass `stm.assert_cov_validity`.
    #Q = stm.enforce_cov_validity(Q)
    #stm.assert_cov_validity(Q)

    dx = ncv.sample_process_noise(dt)
    assert dx.shape == (ncv.dimension,)

    ncv1 = ncv.copy()
    assert ncv1
