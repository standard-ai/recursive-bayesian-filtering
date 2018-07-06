'''
Test `cyclic_interval_wrapping`.

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
import unittest

# Scientific Computing
import numpy as np

# Custom
import cyclic_interval_wrapping as ciwm


def wrap_0_to_2pi_slow(x: float):
    '''
    Map/wrap an angle x into the interval [0.0, 2.0*np.pi) with 0.0 and
    2.0*np.pi identified.

    Args:
        x: angle in radians.

    Returns:
        Angle in [0.0, 2.0*np.pi) in radians.
    '''
    y = float(np.arctan2(np.sin(x), np.cos(x)))
    if y >= 0:
        return y
    return float(y + 2.0*np.pi)  # y < 0


def wrap_mpi_to_pi_slow(x: float):
    '''
    Map/wrap an angle x into the interval [-np.pi, np.pi] with -np.pi and
    np.pi identified.

    Args:
        x: angle in radians.

    Returns:
        Angle in [0.0, 2.0*np.pi), in radians.
    '''
    return float(np.arctan2(np.sin(x), np.cos(x)))


class TestWrapping(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        '''Randomly sample unwrapped angles as test cases.'''
        num_samples = 10000
        interval_radius = 40
        cls.abs_error_max = 0.0000000000001
        cls.samples = []
        for i in range(num_samples):
            if i < num_samples / 2:
                # positive samples
                cls.samples.append(float(np.random.uniform(0.0, interval_radius)))
            else:
                # negative samples
                cls.samples.append(float(np.random.uniform(-interval_radius, 0.0)))

    def test_wrap_0_to_2pi(self):
        '''Test wrapping at `0.0` and `2.0*np.pi`.'''
        failed = False
        for x in self.samples:
            x_mod = ciwm.wrap_0_to_2pi(x)
            x_mod_check = wrap_0_to_2pi_slow(x)
            d = abs(x_mod - x_mod_check)
            if d > self.abs_error_max:
                failed = True
        assert not failed, 'Problem with `wrap_0_to_2pi`!'

    def test_wrap_mpi_to_pi(self):
        '''Test wrapping at `-np.pi` and `np.pi`.'''
        failed = False
        for x in self.samples:
            x_mod = ciwm.wrap_mpi_to_pi(x)
            x_mod_check = wrap_mpi_to_pi_slow(x)
            d = abs(x_mod - x_mod_check)
            if d > self.abs_error_max:
                failed = True
        assert not failed, 'Problem with `wrap_mpi_to_pi`!'


def test_geodesic_difference():
    abs_error_max = 0.0000000000001
    x = ciwm.geodesic_difference(1.23, 4.32)
    assert abs(x) < np.pi + abs_error_max, 'Geodesic difference failure!'


def test_geodesic_distance():
    abs_error_max = 0.0000000000001
    x = ciwm.geodesic_distance(3.45, 1.32)
    assert x < np.pi + abs_error_max, 'Geodesic difference failure!'
