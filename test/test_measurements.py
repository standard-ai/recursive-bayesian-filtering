'''Test `measurements`.

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
# import pytest

# Scientific Computing
import numpy as np

# Custom
import measurements as mm


def test_PositionMeasurement():
    dimension = 3
    time = 0.232
    frame_num = 5
    measurement = mm.PositionMeasurement(
        mean=np.random.random(dimension),
        cov=np.eye(dimension), time=time, frame_num=frame_num)
    assert measurement.dimension == dimension
    x = np.random.random(2*dimension)
    assert measurement(x).shape == (dimension,)
    assert measurement.mean.shape == (dimension,)
    assert measurement.cov.shape == (dimension, dimension)
    assert measurement.time == time
    assert measurement.frame_num == frame_num
    assert measurement.geodesic_difference(
        np.random.random(dimension), np.random.random(dimension)).shape \
        == (dimension,)
    assert measurement.jacobian().shape == (dimension, 2*dimension)
    assert measurement.copy() is not measurement
