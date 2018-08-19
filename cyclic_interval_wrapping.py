'''
Functions for wrapping real numbers into cyclic intervals, especially
angles.

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

# Scientific
import numpy as np


def wrap(x: float, l: float = 1.0) -> float:
    '''
    Map/wrap x into the interval [0.0, l) with 0.0 and l identified.

    Args:
        x: number to map.
        l: length of mod interval.

    Returns:
        The image of x in [0.0, l).
    '''
    return float(np.mod(x, l))


def wrap_0_to_2pi(x: float) -> float:
    '''
    Map/wrap an angle x into the interval [0.0, 2.0*np.pi) with 0.0 and
    2.0*np.pi identified.

    Args:
        x: angle in radians.

    Returns:
        Angle radians in [0.0, 2.0*np.pi).
    '''
    return wrap(x, 2.0*np.pi)


def wrap_mpi_to_pi(x: float) -> float:
    '''
    Map/wrap an angle x into the interval [-np.pi, np.pi) with -np.pi and
    np.pi identified.

    Args:
        x: angle in radians.

    Returns:
        Angle radians in [0.0, 2.0*np.pi).
    '''
    y = wrap(x, 2.0*np.pi)
    if y <= np.pi:
        return y
    return float(y - 2.0*np.pi)  # y > np.pi


def geodesic_difference(a: float, b: float) -> float:
    '''
    Geodesic difference on the circle between angles in radians.

    Args:
        a: angle in radians.
        b: angle in radians.

    Returns:
        Geodesic difference in radians.
    '''
    a_wrapped = wrap_0_to_2pi(a)
    b_wrapped = wrap_0_to_2pi(b)

    d = a_wrapped - b_wrapped
    d_magnitude = abs(d)
    d_alternate_magnitude = abs(2.0*np.pi - d_magnitude)

    if d_alternate_magnitude < d_magnitude:
        return float(-np.sign(d)*d_alternate_magnitude)
    return d


def geodesic_distance(a: float, b: float) -> float:
    '''
    Geodesic distance on the circle between angles in radians.

    Args:
        a: angle in radians.
        b: angle in radians.

    Returns:
        Geodesic distance in radians.
    '''
    a_wrapped = wrap_0_to_2pi(a)
    b_wrapped = wrap_0_to_2pi(b)

    d = a_wrapped - b_wrapped
    d_magnitude = abs(d)
    d_alternate_magnitude = abs(2.0*np.pi - d_magnitude)

    return min(d_magnitude, d_alternate_magnitude)
