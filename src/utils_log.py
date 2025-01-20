import pydantic as pyd
import numpy as np
import scipy.stats as spst
import findiff as fd

import src.general as grl


class TransformationLog(pyd.BaseModel):
    k: float
    sig: float
    mkt: grl.Market

    @property
    def c(self):
        return self.sig**2/2

    @property
    def r_(self):
        return self.mkt.r - self.c

    def x(self, s):
        '''
        x = log(S/K)
        '''
        return np.log(s/self.k)

    def s(self, x):
        '''
        s = K e^x
        '''
        return self.k*np.exp(x)


class EuropeanOptionBS2:
    def __init__(self,
                 k: float,
                 sig: float,
                 mkt: grl.Market):
        self.k = k
        self.sig = sig
        self.mkt = mkt

        self.tfm: TransformationLog = TransformationLog(k=k,
                                                        sig=sig,
                                                        mkt=mkt)

    # @property
    # def c(self):
    #     return self.sig**2/2

    def D(self, th):
        return self.mkt.D(th)

    def F(self, th, s):
        return s/self.D(th)

    def d1(self, th, s):
        return (np.log(self.F(th, s) / self.k) + self.sig**2*th/2)/(self.sig*th**0.5)

    def d2(self, th, s):
        '''
        Q(S_T > K) = 1 - Q(S_T < K)
        S = S_T
        s_T = ln(S_T/S)
        k = ln(K/S)
        Q(S_T < K) = Q(s_T < k)
        = Q(z < [k - (r - c)th]/[sig*th**0.5]) = N(-d2) = 1 - N(d2)
        Q(S_T > K) = N(d2)
        '''
        return self.d1(th, s) - self.sig*th**0.5

    # def d1(self, th, s):
    #     return self.d2(th, s) + self.sig*th**0.5

    def call(self, th, s):
        N = spst.norm.cdf
        n1 = N(self.d1(th, s))
        n2 = N(self.d2(th, s))
        return self.D(th)*(self.F(th, s)*n1 - self.k*n2)

    def put(self, th, s):
        N = spst.norm.cdf
        n1 = N(-self.d1(th, s))
        n2 = N(-self.d2(th, s))
        return self.mkt.D(th)*(self.k*n2 - self.F(th, s)*n1)


class Lattice:
    def __init__(self, nx: int, nt: int, T: float, tfm: TransformationLog, alpha: float = 0.01) -> None:
        self.nx = nx
        self.nt = nt

        z_a = spst.norm().ppf((1 - alpha/2))
        x_min = - T**0.5*tfm.sig*z_a
        x_max = tfm.r_*T + T**0.5*tfm.sig*z_a
        self.x = np.linspace(x_min, x_max, nx)

        self.t = np.linspace(0, T, nt)

        self.dx = self.x[1] - self.x[0]
        self.dt = self.t[1] - self.t[0]

    def meshgrid_tx(self) -> list:
        return np.meshgrid(self.t, self.x, indexing='ij')

    def empty_tx(self):
        return np.empty((self.nt, self.nx))

    def empty_t(self):
        return np.empty(self.nt)

    def empty_x(self):
        return (np.empty(self.nx))
