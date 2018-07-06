'''
Target dynamic models with a unified interface for use in recursive
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

# Scientific Computing
import numpy as np

# Custom
import stats_tools as stm


# -----  Interfaces -----


class DynamicModel(ABC):
    '''
    Dynamic model interface.

    Args:
        dimension: native state dimension.
        dimension_pv: PV state dimension.
        num_process_noise_parameters: process noise parameter space dimension.
          This for UKF applications. Can be left as `None` for EKF and most
          other filters.
    '''
    def __init__(
            self, dimension: int,
            dimension_pv: int,
            num_process_noise_parameters: int = None) -> None:
        self._dimension = dimension
        self._dimension_pv = dimension_pv
        self._num_process_noise_parameters = num_process_noise_parameters

    @property
    def dimension(self) -> int:
        '''Native state dimension access.'''
        return self._dimension

    @property
    def dimension_pv(self) -> int:
        '''PV state dimension access.'''
        return self._dimension_pv

    @property
    def num_process_noise_parameters(self) -> Optional[int]:
        '''Process noise parameters space dimension access.'''
        return self._num_process_noise_parameters

    @abstractmethod
    def __call__(
            self,
            x: np.ndarray, dt: float,
            do_normalization: bool = True) -> np.ndarray:
        '''
        Integrate native state `x` over time interval `dt`.

        Args:
            x: current native state. If the DynamicModel is non-differentiable,
              be sure to handle the case of `x` being augmented with process
              noise parameters.
            dt: time interval to integrate over.
            do_normalization: whether to perform normalization on output, e.g.,
              mod'ing angles into an interval.

        Returns:
            Native state x integrated dt into the future.
        '''
        pass

    def geodesic_difference(
            self, x1: np.ndarray, x0: np.ndarray) -> np.ndarray:
        '''
        Compute and return the geodesic difference between 2 native states.
        This is a generalization of the Euclidean operation `x1 - x0`.

        Args:
            x1: native state.
            x0: native state.

        Returns:
            Geodesic difference between native states `x1` and `x2`.
        '''
        return x1 - x0  # Default to Euclidean behavior.

    @abstractmethod
    def mean2pv(self, x: np.ndarray) -> np.ndarray:
        '''
        Compute and return PV state from native state. Useful for combining
        state estimates of different types in IMM (Interacting Multiple Model)
        filtering.

        ::CAUTION:: For efficiency, may return a reference to the input.
        Deepcopy as necessary to prevent unexpected changes.

        Args:
            x: native state estimate mean.

        Returns:
            PV state estimate mean.
        '''
        pass

    @abstractmethod
    def cov2pv(self, P: np.ndarray) -> np.ndarray:
        '''
        Compute and return PV covariance from native covariance. Useful for
        combining state estimates of different types in IMM (Interacting
        Multiple Model) filtering.

        ::CAUTION:: For efficiency, may return a reference to the input.
        Deepcopy as necessary to prevent unexpected changes.

        Args:
            P: native state estimate covariance.

        Returns:
            PV state estimate covariance.
        '''
        pass

    @abstractmethod
    def process_noise_cov(self, dt: float = 0.0) -> np.ndarray:
        '''
        Compute and return process noise covariance (Q).

        Args:
            dt: time interval to integrate over.

        Returns:
            Read-only covariance (Q). For a DifferentiableDynamicModel, this is
            the covariance of the native state `x` resulting from stochastic
            integration (for use with EKF). Otherwise, it is the covariance
            directly of the process noise parameters (for use with UKF).
        '''
        pass

    def sample_process_noise(self, dt: float = 0.0) -> np.ndarray:
        '''
        Sample and return a state displacement from the process noise
        distribution over a time interval.

        Args:
            dt: time interval that process noise accumulates over.

        Returns:
            State displacement.
        '''
        Q = self.process_noise_cov(dt)
        dx = stm.sample_from_normal_distribution(cov=Q)
        return dx.flatten()

    @abstractmethod
    def copy(self) -> 'DynamicModel':
        '''Deepcopy'''
        pass


class DifferentiableDynamicModel(DynamicModel):
    '''
    DynamicModel for which state transition Jacobians can be efficiently
    calculated, usu. analytically or by automatic differentiation.
    '''
    @abstractmethod
    def jacobian(self, dt: float) -> np.ndarray:
        '''
        Compute and return native state transition Jacobian (F) over time
        interval `dt`.

        Args:
            dt: time interval to integrate over.

        Returns:
            Read-only Jacobian (F) of integration map (f).
        '''
        pass


class Ncp(DifferentiableDynamicModel):
    '''
    NCP (Nearly-Constant Position) dynamic model. May be subclassed, e.g., with
    CWNV (Continuous White Noise Velocity) or DWNV (Discrete White Noise
    Velocity).

    Args:
        dimension: native state dimension.
        sv2: variance of velocity. Usually chosen so that the standard
          deviation is roughly half of the max velocity one would ever expect
          to observe.
    '''
    def __init__(self, dimension: int, sv2: float) -> None:
        dimension_pv = 2*dimension
        super().__init__(
            dimension, dimension_pv, num_process_noise_parameters=1)
        self._sv2 = sv2
        self._F_cache = np.eye(dimension)  # State transition matrix cache
        self._F_cache.flags.writeable = False
        self._Q_cache = {}  # Process noise cov cache

    def __call__(
            self, x: np.ndarray, dt: float,
            do_normalization: bool = True) -> np.ndarray:
        '''
        Integrate native state `x` over time interval `dt`.

        Args:
            x: current native state. If the DynamicModel is non-differentiable,
              be sure to handle the case of `x` being augmented with process
              noise parameters.
            dt: time interval to integrate over.
            do_normalization: whether to perform normalization on output, e.g.,
              mod'ing angles into an interval. Has no effect for this subclass.

        Returns:
            Native state x integrated dt into the future.
        '''
        return x.copy()

    def mean2pv(self, x: np.ndarray) -> np.ndarray:
        '''
        Compute and return PV state from native state. Useful for combining
        state estimates of different types in IMM (Interacting Multiple Model)
        filtering.

        Args:
            x: native state estimate mean.

        Returns:
            PV state estimate mean.
        '''
        x_pv = np.zeros(2*self._dimension)
        x_pv[:self._dimension] = x
        return x_pv

    def cov2pv(self, P: np.ndarray) -> np.ndarray:
        '''
        Compute and return PV covariance from native covariance. Useful for
        combining state estimates of different types in IMM (Interacting
        Multiple Model) filtering.

        Args:
            P: native state estimate covariance.

        Returns:
            PV state estimate covariance.
        '''
        d = 2*self._dimension
        P_pv = np.zeros((d, d))
        P_pv[:self._dimension, :self._dimension] = P
        return P_pv

    def jacobian(self, dt: float) -> np.ndarray:
        '''
        Compute and return cached native state transition Jacobian (F) over
        time interval `dt`.

        Args:
            dt: time interval to integrate over.

        Returns:
            Read-only Jacobian (F) of integration map (f).
        '''
        return self._F_cache

    @abstractmethod
    def process_noise_cov(self, dt: float = 0.0) -> np.ndarray:
        '''
        Compute and return cached process noise covariance (Q).

        Args:
            dt: time interval to integrate over.

        Returns:
            Read-only covariance (Q) of the native state `x` resulting from
            stochastic integration (for use with EKF).
        '''
        pass


class Ncv(DifferentiableDynamicModel):
    '''
    NCV (Nearly-Constant Velocity) dynamic model. May be subclassed, e.g., with
    CWNA (Continuous White Noise Acceleration) or DWNA (Discrete White Noise
    Acceleration).

    Args:
        dimension: native state dimension.
        sa2: variance of acceleration. Usually chosen so that the standard
          deviation is roughly half of the max acceleration one would ever
          expect to observe.
    '''
    def __init__(self, dimension: int, sa2: float) -> None:
        dimension_pv = dimension
        super().__init__(
            dimension, dimension_pv, num_process_noise_parameters=1)
        self._sa2 = sa2
        self._F_cache = {}  # State transition matrix cache
        self._Q_cache = {}  # Process noise cov cache

    def __call__(
            self, x: np.ndarray, dt: float,
            do_normalization: bool = True) -> np.ndarray:
        '''
        Integrate native state `x` over time interval `dt`.

        Args:
            x: current native state. If the DynamicModel is non-differentiable,
              be sure to handle the case of `x` being augmented with process
              noise parameters.
            dt: time interval to integrate over.
            do_normalization: whether to perform normalization on output, e.g.,
              mod'ing angles into an interval. Has no effect for this subclass.

        Returns:
            Native state x integrated dt into the future.
        '''
        F = self.jacobian(dt)
        return F.dot(x)

    def mean2pv(self, x: np.ndarray) -> np.ndarray:
        '''
        Compute and return PV state from native state. Useful for combining
        state estimates of different types in IMM (Interacting Multiple Model)
        filtering.

        ::CAUTION:: For efficiency, returns a reference to the input. Deepcopy
        as necessary to prevent unexpected changes.

        Args:
            x: native state estimate mean.

        Returns:
            PV state estimate mean.
        '''
        return x

    def cov2pv(self, P: np.ndarray) -> np.ndarray:
        '''
        Compute and return PV covariance from native covariance. Useful for
        combining state estimates of different types in IMM (Interacting
        Multiple Model) filtering.

        ::CAUTION:: For efficiency, returns a reference to the input. Deepcopy
        as necessary to prevent unexpected changes.

        Args:
            P: native state estimate covariance.

        Returns:
            PV state estimate covariance.
        '''
        return P

    def jacobian(self, dt: float) -> np.ndarray:
        '''
        Compute and return cached native state transition Jacobian (F) over
        time interval `dt`.

        Args:
            dt: time interval to integrate over.

        Returns:
            Read-only Jacobian (F) of integration map (f).
        '''
        if dt not in self._F_cache:
            d = self._dimension
            F = np.eye(d)
            F[:d//2, d//2:] = dt*np.eye(d//2)
            F.flags.writeable = False
            self._F_cache[dt] = F

        return self._F_cache[dt]

    @abstractmethod
    def process_noise_cov(self, dt: float = 0.0) -> np.ndarray:
        '''
        Compute and return cached process noise covariance (Q).

        Args:
            dt: time interval to integrate over.

        Returns:
            Read-only covariance (Q) of the native state `x` resulting from
            stochastic integration (for use with EKF).
        '''
        pass




# ----- Concrete Classes -----


class NcpContinuous(Ncp):
    '''
    NCP (Nearly-Constant Position) dynamic model with CWNV (Continuous White
    Noise Velocity).

    References:
        "Estimation with Applications to Tracking and Navigation" by Y. Bar-
        Shalom et al, 2001, p.269.

    Args:
        dimension: native state dimension.
        sv2: variance of velocity. Usually chosen so that the standard
          deviation is roughly half of the max velocity one would ever expect
          to observe.
    '''
    def process_noise_cov(self, dt: float = 0.0) -> np.ndarray:
        '''
        Compute and return cached process noise covariance (Q).

        Args:
            dt: time interval to integrate over.

        Returns:
            Read-only covariance (Q) of the native state `x` resulting from
            stochastic integration (for use with EKF).
        '''
        if dt not in self._Q_cache:
            # q: continuous-time process noise intensity with units
            #   length^2/time (m^2/s). Choose `q` so that changes in position,
            #   over a sampling period `dt`, are roughly `sqrt(q*dt)`.
            q = self._sv2*dt
            Q = q*dt*np.eye(self._dimension)
            Q.flags.writeable = False
            self._Q_cache[dt] = Q

        return self._Q_cache[dt]

    def copy(self) -> 'NcpContinuous':
        '''Deepcopy, except does not copy cached data.'''
        return NcpContinuous(self._dimension, self._sv2)


class NcvContinuous(Ncv):
    '''
    NCV (Nearly-Constant Velocity) dynamic model with CWNA (Continuous White
    Noise Acceleration).

    References:
        "Estimation with Applications to Tracking and Navigation" by Y. Bar-
        Shalom et al, 2001, p.269.

    Args:
        dimension: native state dimension.
        sa2: variance of acceleration. Usually chosen so that the standard
          deviation is roughly half of the max acceleration one would ever
          expect to observe.
    '''
    def process_noise_cov(self, dt: float = 0.0) -> np.ndarray:
        '''
        Compute and return cached process noise covariance (Q).

        Args:
            dt: time interval to integrate over.

        Returns:
            Read-only covariance (Q) of the native state `x` resulting from
            stochastic integration (for use with EKF).
        '''
        if dt not in self._Q_cache:
            # q: continuous-time process noise intensity with units
            #   length^2/time^3 (m^2/s^3). Choose `q` so that changes in
            #   velocity, over a sampling period `dt`, are roughly
            #   `sqrt(q*dt)`.
            q = self._sa2*dt
            d = self._dimension
            dt2 = dt*dt
            dt3 = dt2*dt
            Q = np.zeros((d, d))
            Q[:d//2, :d//2] = dt3*np.eye(d//2)/3.0
            Q[:d//2, d//2:] = dt2*np.eye(d//2)/2.0
            Q[d//2:, :d//2] = dt2*np.eye(d//2)/2.0
            Q[d//2:, d//2:] = dt*np.eye(d//2)
            Q *= q
            Q.flags.writeable = False
            self._Q_cache[dt] = Q

        return self._Q_cache[dt]

    def copy(self) -> 'NcvContinuous':
        '''Deepcopy, except does not copy cached data.'''
        return NcvContinuous(self._dimension, self._sa2)


class NcpDiscrete(Ncp):
    '''
    NCP (Nearly-Constant Position) dynamic model with DWNV (Discrete White
    Noise Velocity).

    References:
        "Estimation with Applications to Tracking and Navigation" by Y. Bar-
        Shalom et al, 2001, p.273.

    Args:
        dimension: native state dimension.
        sv2: variance of velocity. Usually chosen so that the standard
          deviation is roughly half of the max velocity one would ever expect
          to observe.
    '''
    def process_noise_cov(self, dt: float = 0.0) -> np.ndarray:
        '''
        Compute and return cached process noise covariance (Q).

        Args:
            dt: time interval to integrate over.

        Returns:
            Read-only covariance (Q) of the native state `x` resulting from
            stochastic integration (for use with EKF).
        '''
        if dt not in self._Q_cache:
            Q = self._sv2*dt*dt*np.eye(self._dimension)
            Q.flags.writeable = False
            self._Q_cache[dt] = Q

        return self._Q_cache[dt]

    def copy(self) -> 'NcpDiscrete':
        '''Deepcopy, except does not copy cached data.'''
        return NcpDiscrete(self._dimension, self._sv2)


class NcvDiscrete(Ncv):
    '''
    NCV (Nearly-Constant Velocity) dynamic model with DWNA (Discrete White
    Noise Acceleration).

    References:
        "Estimation with Applications to Tracking and Navigation" by Y. Bar-
        Shalom et al, 2001, p.273.

    Args:
        dimension: native state dimension.
        sa2: variance of acceleration. Usually chosen so that the standard
          deviation is roughly half of the max acceleration one would ever
          expect to observe.
    '''
    def process_noise_cov(self, dt: float = 0.0) -> np.ndarray:
        '''
        Compute and return cached process noise covariance (Q).

        Args:
            dt: time interval to integrate over.

        Returns:
            Read-only covariance (Q) of the native state `x` resulting from
            stochastic integration (for use with EKF). Mind that this Q, modulo
            numerical error, has rank `dimension/2`. So, it is only positive
            semi-definite.
        '''
        if dt not in self._Q_cache:
            d = self._dimension
            dt2 = dt*dt
            dt3 = dt2*dt
            dt4 = dt2*dt2
            Q = np.zeros((d, d))
            Q[:d//2, :d//2] = 0.25*dt4*np.eye(d//2)
            Q[:d//2, d//2:] = 0.5*dt3*np.eye(d//2)
            Q[d//2:, :d//2] = 0.5*dt3*np.eye(d//2)
            Q[d//2:, d//2:] = dt2*np.eye(d//2)
            Q *= self._sa2
            Q.flags.writeable = False
            self._Q_cache[dt] = Q

        return self._Q_cache[dt]

    def copy(self) -> 'NcvDiscrete':
        '''Deepcopy, except does not copy cached data.'''
        return NcvDiscrete(self._dimension, self._sa2)