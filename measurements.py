'''
Gaussian measurement models with a unified interface for use in recursive
Bayesian filtering.

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

# pylint: disable=W0611, R0201, W0613, E0602

# Standard
from typing import Optional
from abc import ABC, abstractmethod

# Scientific
import numpy as np

# Custom
import stats_tools as stm


# -----  Interfaces -----


class Measurement(ABC):
    '''
    Gaussian measurement interface.

    Instance Variables:
        _dimension: dimension of measurement space.
        _mean: mean of measurement.
        _cov: covariance of measurement.
        _time: optional continuous time of measurement. If this is not set,
          `_frame_num` must be.
        _frame_num: optional discrete time of measurement. If this is not set,
          `_time` must be.

    Args:
        mean: mean of measurement distribution.
        cov: covariance of measurement distribution.
        time: optional continuous time of measurement. If this is not provided,
          `frame_num` must be.
        frame_num: optional discrete time of measurement. If this is not
          provided, `time` must be.
    '''
    def __init__(
            self, mean: np.ndarray, cov: np.ndarray,
            time: float = None, frame_num: int = None):
        self._dimension = len(mean)
        self._mean = mean.copy()
        self._mean.flags.writeable = False
        self._cov = cov.copy()
        self._cov.flags.writeable = False

        assert not (time is None and frame_num is None), \
            'Must provided either `time` or `frame_num`!'
        self._time = time
        self._frame_num = frame_num

    @property
    def dimension(self) -> int:
        '''Measurement space dimension access.'''
        return self._dimension

    @property
    def mean(self) -> np.ndarray:
        '''Measurement mean (`z` in most Kalman Filtering literature).'''
        return self._mean

    @property
    def cov(self) -> np.ndarray:
        '''Noise covariance (`R` in most Kalman Filtering literature).'''
        return self._cov

    @property
    def time(self) -> Optional[float]:
        '''Continuous measurement time access.'''
        return self._time

    @property
    def frame_num(self) -> Optional[int]:
        '''Discrete measurement time access.'''
        return self._frame_num

    @abstractmethod
    def __call__(
            self, x: np.ndarray, do_normalization: bool = True) -> np.ndarray:
        '''
        Measurement map (h) for predicting a measurement `z` from target
        state `x`.

        Args:
            x: PV state.
            do_normalization: whether to normalize output, e.g., mod'ing angles
              into an interval.

        Returns:
            Measurement predicted from state `x`.
        '''
        pass

    def geodesic_difference(
            self, z1: np.ndarray, z0: np.ndarray) -> np.ndarray:
        '''
        Compute and return the geodesic difference between 2 measurements.
        This is a generalization of the Euclidean operation `z1 - z0`.

        Args:
            z1: measurement.
            z0: measurement.

        Returns:
            Geodesic difference between `z1` and `z2`.
        '''
        return z1 - z0  # Default to Euclidean behavior.

    @abstractmethod
    def copy(self) -> 'Measurement':
        '''Deepcopy.'''
        pass


class DifferentiableMeasurement(Measurement):
    '''
    Interface for Gaussian measurement for which Jacobians can be efficiently
    calculated, usu. analytically or by automatic differentiation.

    Instance Variables (in addition to those of `Measurement`):
        _jacobian: Jacobian matrix of the map from PV space to P space.
    '''
    @abstractmethod
    def jacobian(self, x: np.ndarray = None) -> np.ndarray:
        '''
        Compute and return Jacobian (H) of measurement map (h) at target PV
        state `x` .

        Args:
            x: PV state. Use default argument `None` when the Jacobian is not
              state-dependent.

        Returns:
            Read-only Jacobian (H) of measurement map (h).
        '''
        pass




# ----- Concrete Classes -----


class PositionMeasurement(DifferentiableMeasurement):
    '''
    Full-rank Gaussian position measurement in Euclidean space.

    Args:
        mean: mean of measurement distribution.
        cov: covariance of measurement distribution.
        time: optional continuous time of measurement. If this is not provided,
          `frame_num` must be.
        frame_num: optional discrete time of measurement. If this is not
          provided, `time` must be.
    '''
    def __init__(
            self, mean: np.ndarray, cov: np.ndarray,
            time: float = None, frame_num: int = None):
        super().__init__(mean, cov, time, frame_num)
        self._jacobian = np.hstack((
            np.eye(self.dimension),
            np.zeros((self.dimension, self.dimension))))
        self._jacobian.flags.writeable = False

    def __call__(
            self, x: np.ndarray, do_normalization: bool = True) -> np.ndarray:
        '''
        Measurement map (h) for predicting a measurement `z` from target
        state `x`.

        Args:
            x: PV state.
            do_normalization: whether to normalize output. Has no effect for
              this subclass.

        Returns:
            Measurement predicted from state `x`.
        '''
        return x[:self._dimension].copy()

    def jacobian(self, x: np.ndarray = None) -> np.ndarray:
        '''
        Compute and return Jacobian (H) of measurement map (h) at target PV
        state `x`.

        Args:
            x: PV state. The default argument `None` may be used in this
              subclass since the Jacobian is not state-dependent.

        Returns:
            Read-only Jacobian (H) of measurement map (h).
        '''
        return self._jacobian

    def copy(self) -> 'PositionMeasurement':
        '''Deepcopy.'''
        return PositionMeasurement(
            self._mean, self._cov, self._time, self._frame_num)
