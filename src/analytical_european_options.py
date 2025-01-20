import numpy as np
import pydantic as pyd
import scipy.stats as spst


class EuropeanoptionsBS(pyd.BaseModel):
    k: float
    sig: float
    r: float
    S: float

    @property
    def c(self):
        return self.sig**2/2

    def D(self, theta: float):
        return np.exp(-self.r*theta)

    def F(self, s: float):
        return s/self.D

    def d1(self, s: float, theta: float):
        return (np.log(self.F(s)/self.k) + self.c*theta)/(self.sig*theta**0.5)

    def d2(self, s: float, theta: float):
        return self.d1(s, theta) - self.sig*theta**0.5

    def call(self, s: float, theta: float):
        N = spst.norm.cdf
        n1 = N(self.d1(s, theta))
        n2 = N(self.d2(s, theta))
        return self.D(theta)*(self.F(s)*n1 - self.k*n2)

    def put(self, s: float, theta: float):
        N = spst.norm.cdf
        n1 = N(-self.d1(s, theta))
        n2 = N(-self.d2(s, theta))
        return self.D(theta)*(self.k*n2 - self.F(s)*n1)
