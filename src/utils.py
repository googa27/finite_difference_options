import numpy as np
import pydantic as pyd
import scipy.stats as spst
import findiff as fd
import scipy.sparse.linalg as spla
import abc
import dataclasses as dtc


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


class TransformationHeat(pyd.BaseModel):
    k: float
    sig: float
    mkt: Market

    @property
    def c(self):
        return self.sig**2/2

    @property
    def mu(self):
        return self.mkt.r - self.c

    def x(self, th, s):
        '''
        x = log(S/K) + mu*th
        '''
        return np.log(s/self.k) + self.mu*th

    def s(self, th, x):
        '''
        s = K e^(x - mu*th)
        '''
        return self.k*np.exp(x - self.mu*th)

    # def psi_call(self, th, x):
    #     z = x - self.c*th
    #     return self.k*np.maximum(np.exp(z) - 1, 0)

    # def psi_put(self, th, x):
    #     z = x - self.c*th
    #     return self.k*np.maximum(1 - np.exp(z), 0)


class EuropeanOption(metaclass=abc.ABCMeta):
    def __init__(self, k: float, sig: float, mkt: Market):
        self.k: float = k
        self.sig: float = sig
        self.mkt: Market = mkt

        self.tfm: TransformationHeat = TransformationHeat(
            k=k, sig=sig, mkt=mkt)

    @abc.abstractmethod
    def payoff(self, s):
        '''
        Payoff of option.
        '''
        pass

    def bc(self, th, s):
        D = self.mkt.D(th)
        B = self.mkt.B(th)
        return D*self.payoff(B*s)

    def bc_heat(self, th, x):
        s = self.tfm.s(th, x)
        B = self.mkt.B(th)
        return B*self.bc(th, s)


class EuropeanCall(EuropeanOption):
    def payoff(self, s):
        return np.maximum(s - self.k, 0)


class EuropeanPut(EuropeanOption):
    def payoff(self, s):
        return np.maximum(self.k - s, 0)

######
######
######


class EuropeanOptionBS:
    def __init__(self,
                 k: float,
                 sig: float,
                 mkt: Market):
        self.k = k
        self.sig = sig
        self.mkt = mkt

        self.tfm: TransformationHeat = TransformationHeat(k=k,
                                                          sig=sig,
                                                          mkt=mkt)

    # @property
    # def c(self):
    #     return self.sig**2/2

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
        return self.tfm.x(th, s)/(self.sig*th**0.5)

    def d1(self, th, s):
        return self.d2(th, s) + self.sig*th**0.5

    def call(self, th, s):
        N = spst.norm.cdf
        n1 = N(self.d1(th, s))
        n2 = N(self.d2(th, s))
        return s*n1 - self.mkt.D(th)*self.k*n2

    def put(self, th, s):
        N = spst.norm.cdf
        n1 = N(-self.d1(th, s))
        n2 = N(-self.d2(th, s))
        return self.mkt.D(th)*self.k*n2 - s*n1


class EuropeanOptionBS2:
    def __init__(self,
                 k: float,
                 sig: float,
                 mkt: Market):
        self.k = k
        self.sig = sig
        self.mkt = mkt

        self.tfm: TransformationHeat = TransformationHeat(k=k,
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
    def __init__(self, nx: int, nt: int, T: float, x_bounds: tuple) -> None:
        self.nx = nx
        self.nt = nt

        self.x = np.linspace(x_bounds[0], x_bounds[1], nx)
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


class FinDiffOption:
    '''
    There's something wrong with the implementation, must be verified through comparison with the analytical solution for european options.
    '''

    def __init__(self, lat: Lattice, sig: float, mkt: Market, k: float):
        self.lat = lat
        self.sig = sig
        self.mkt = mkt

        self.ec = EuropeanCall(k, sig, mkt)
        self.ep = EuropeanPut(k, sig, mkt)
        self.tfm = TransformationHeat(
            k=k, sig=sig, mkt=mkt)

    @property
    def c(self):
        return self.tfm.c

    @property
    def k_speed(self):
        return self.mkt.r/self.c

    def op_lbs(self):
        d_dx = fd.FinDiff(0, self.lat.dx, 1)
        d_dx2 = fd.FinDiff(0, self.lat.dx, 2)
        I = fd.Identity()

        return self.c*d_dx2 + self.tfm.mu*d_dx - self.mkt.r*I

    def op_lbs2(self):
        d_dx = fd.FinDiff(0, self.lat.dx, 1)
        d_dx2 = fd.FinDiff(0, self.lat.dx, 2)

        return self.c*d_dx2 + self.tfm.mu*d_dx

    def op_heat(self):
        d_dx2 = fd.FinDiff(0, self.lat.dx, 2)

        return self.c*d_dx2

    def evolution_op_lbs(self):
        '''
        e^(dt*A)
        '''
        A = self.op_lbs()
        M_step = self.lat.dt*A.matrix(self.lat.x.shape)
        return spla.expm(M_step)

    def evolution_op_heat(self):
        A = self.op_heat()
        M_step = self.lat.dt*A.matrix(self.lat.x.shape)
        return spla.expm(M_step)

    def solve_heat(self, call: bool, american: bool):

        # Setup of Evolution Operator th -> th + d_th

        Evo = self.evolution_op_heat()

        # theta = 1 # 0 = explicit, 1 = implicit, for crank-nicholson (here we just use exponentials)

        u_tx = self.lat.empty_tx()
        x_ex = self.lat.empty_t()

        # INITIAL CONDITIONS

        s = self.tfm.s(th=0, x=self.lat.x)

        if call:
            payoff_v = self.ec.payoff(s)

        else:
            payoff_v = self.ep.payoff(s)

        u_tx[0] = payoff_v

        # MAIN PROCEDURE

        for i, ti in enumerate(self.lat.t[:-1]):
            # u_tx[i, 0] = 2*u_tx[i, 1] - u_tx[i, 2]
            # u_tx[i, -1] = 2*u_tx[i, -2] - u_tx[i, -3]

            # u_tx[i, 0] = u_tx[i, 1] - 1 * \
            #     self.lat.dx/self.mkt.D(ti)*(1 - call)
            # u_tx[i, -1] = u_tx[i, -2] + \
            #     1*self.lat.dx/self.mkt.D(ti)*call

            u_aux = Evo@u_tx[i]
            # # IMPLEMENT BOUNDARY CONDITIONS
            # u_aux[0] = 2*u_aux[1] - u_aux[2]
            # u_aux[-1] = 2*u_aux[-1] - u_aux[-2]

            payoff_u = payoff_v/self.mkt.D(ti)

            if american:
                v_x = np.maximum(u_aux, payoff_u)
                u_tx[i + 1] = v_x

                # exercise boundary calculation
                if call:
                    x_ex[i + 1] = self.lat.x[np.argmin(u_aux < payoff_u)]
                else:
                    x_ex[i + 1] = self.lat.x[np.argmax(u_aux > payoff_u)]
            else:
                u_tx[i + 1] = u_aux

            # u_tx[i + 1, 0] = 2*u_tx[i + 1, 1] - u_tx[i + 1, 2]
            # u_tx[i + 1, -1] = 2*u_tx[i + 1, -2] - u_tx[i + 1, -3]

            # u_tx[i + 1, 0] = u_tx[i + 1, 1] - 1 * \
            #     self.lat.dx/self.mkt.D(ti)*(1 - call)
            # u_tx[i + 1, -1] = u_tx[i + 1, -2] + \
            #     1*self.lat.dx/self.mkt.D(ti)*call

            # if call:
            #     u_tx[i + 1, 0] = 0
            #     u_tx[i + 1, -1] = (self.tfm.s(ti, self.lat.x[-1]
            #                                   ) - self.tfm.k)/self.mkt.D(ti)
            # u_tx[i + 1, -1] = (self.tfm.s(ti, self.lat.x[-1]
            #                               ) - self.tfm.k)

            # u_tx[i + 1, 0] = u_tx[i + 1, 1] - \
            #     self.lat.dx*1 / self.mkt.D(ti)*(1 - call)
            # u_tx[i + 1, -1] = u_tx[i + 1, -2] + \
            #     0.35*1/self.mkt.D(ti)*call
        # change from x -> s

        s_ex = self.tfm.s(self.lat.t, x_ex)

        # present value transformation adjustment

        tt, _ = self.lat.meshgrid_tx()
        u_tx *= self.mkt.D(tt)

        # go from tenor to time

        u_tx = u_tx[::-1]
        s_ex = s_ex[::-1]

        return u_tx, s_ex
