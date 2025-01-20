import abc
import pydantic as pyd
import findiff as fd
import numpy as np


class Underlying(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def generator(self):
        pass


class UnderlyingGBM(pyd.BaseModel, Underlying):
    mu: float
    sig: float

    @property
    def c(self):
        return self.sig**2/2

    def generator(self, s: np.ndarray):
        ds = s[1] - s[0]

        d_ds = fd.FinDiff(0, ds, 1)
        d_ds2 = fd.FinDiff(0, ds, 2)

        d1 = fd.Coef(s)*d_ds
        d2 = fd.Coef(s**2)*d_ds2

        return self.c*d2 + self.mu*d1

    def log_generator(self, y: np.ndarray):
        dy = y[1] - y[0]

        d_dy = fd.FinDiff(0, dy, 1)
        d_dy2 = fd.FinDiff(0, dy, 2)

        return self.c*d_dy2 + (self.mu - self.c)*d_dy

    def heat_generator(self, x: np.ndarray):
        dx = x[1] - x[0]

        d_dx2 = fd.FinDiff(0, dx, 2)

        return self.c*d_dx2


class UnderlyingHeston(pyd.BaseModel, Underlying):
    '''
    UNUSED
    '''
    mu: float
    a: float
    b: float
    c: float


class Derivative(pyd.BaseModel):
    model_config = pyd.ConfigDict(arbitrary_types_allowed=True)

    uly: Underlying
    T: float


class Market(pyd.BaseModel):
    r: float

    def D(self, th):
        return np.exp(-self.r*th)

    def B(self, th):
        return self.D(-th)
