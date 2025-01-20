import streamlit as st
import matplotlib.pyplot as plt
import matplotlib as mpl
import findiff as fd
import numpy as np
import pydantic as pyd
import scipy.stats as spst
import seaborn as sns

import config.CONFIG as CFG

with st.sidebar:
    is_american = st.checkbox('american')
    is_call = st.checkbox('call', value=True)
    left_bc = st.radio('left boundary conditions',
                       ['dirichlet',
                        'neumann',
                        'gamma']
                       )
    right_bc = st.radio('right boundary conditions',
                        ['dirichlet',
                         'neumann',
                         'gamma']
                        )

    nx = st.slider('nx', 1, 1000, 150)
    nt = st.slider('nt', 1, 1000, 100)

    T = st.slider('Maturity', 1., 30., 5.)

    r = st.slider('r', -0.1, 0.5, 0.03)
    k = st.slider('k', 0.1, 2., 0.4)
    sig = st.slider(r'$\sigma$', 0., 0.8, 0.2)

    c = sig**2/2
    r_adj = r - c

    la_ = st.slider('log alpha',
                    -3.,
                    -1.,
                    -2.)

    a_ = 10**la_
    z_a_ = spst.norm().ppf(1 - a_)
    x_min = - sig*T**0.5*z_a_ - (r + c)*T
    x_max = sig*T**0.5*z_a_ - (r - c)*T

    # s_max = k*np.exp(x_max)
    lam = st.slider('lambda', 0., 1., 0.5)


class Market(pyd.BaseModel):
    r: float

    def D(self, th: float):
        return np.exp(-self.r*th)


class TransformerLog(pyd.BaseModel):
    k: float

    def s(self, x: float):
        return self.k*np.exp(x)

    def s_x(self, x: float):
        return self.s(x)

    def s_xx(self, x: float):
        return self.s(x)

    def x(self, s: float):
        return np.log(s/self.k)

    def x_s(self, s: float):
        return 1/s

    def x_ss(self, s: float):
        return -1/s**2


tfm = TransformerLog(k=k)
mkt = Market(r=r)

t = np.linspace(0, T, nt)
x = np.linspace(x_min, x_max, nx)
s = tfm.s(x)
dt = t[1] - t[0]
dx = x[1] - x[0]

L = (c * fd.FinDiff(0, dx, 2)
     + r_adj*fd.FinDiff(0, dx, 1)
     - r*fd.Identity()
     )
A = fd.Identity() - dt*lam*L
B = fd.Identity() + dt*(1 - lam)*L

if is_call:
    phi = np.maximum(s - k, 0)
else:  # put
    phi = np.maximum(k - s, 0)

v_tx = np.empty((nt, nx))
v_tx[0] = phi
s_ex = np.empty((nt, ))


def get_bcs(x,
            s,
            bc_types: tuple[str, str] = ['gamma', 'gamma'],
            is_call: bool = True):
    # gamma only for the moment
    dx = x[1] - x[0]
    bc = fd.BoundaryConditions(x.shape)
    d1 = fd.FinDiff(0, dx, 1)
    d2 = fd.FinDiff(0, dx, 2)

    # LEFT CONDITIONS

    if bc_types[0] == 'dirichlet':
        if not is_call:
            raise NotImplementedError(bc_types[0])
        bc[0] = 0
    if bc_types[0] == 'neumann':
        bc[0] = d1, -s[0]*(not is_call)
    elif bc_types[0] == 'gamma':
        bc[0] = d2, -s[0]*(not is_call)

    # RIGHT CONDITIONS

    if bc_types[-1] == 'dirichlet':
        if is_call:
            raise NotImplementedError(bc_types[1])
        bc[-1] = 0
    if bc_types[-1] == 'neumann':
        bc[-1] = d1, s[-1]*is_call
    if bc_types[-1] == 'gamma':
        bc[-1] = d2, s[-1]*is_call

    return bc


bc = get_bcs(x,
             s,
             bc_types=[left_bc, right_bc],
             is_call=is_call)

for i, ti in enumerate(t[:-1]):
    pde = fd.PDE(lhs=A,
                 rhs=B(v_tx[i]),
                 bcs=bc)

    v_tx[i+1] = pde.solve()
    if is_american:
        v_tx[i + 1] = np.maximum(v_tx[i + 1], phi)
        if is_call:
            s_ex[i + 1] = s[np.argmin(v_tx[i + 1] > phi)]
        else:  # put
            s_ex[i + 1] = s[np.argmax(v_tx[i + 1] > phi)]

## PLOT ##


def plot_1d(x, y,
            title: str,
            label: str = 'option',
            zero_line: bool = True) -> None:
    fig, ax = plt.subplots()
    if isinstance(y, list):
        assert isinstance(label, list)
        for y_i, label_i in zip(y, label):
            ax.plot(x, y_i, label=label_i)
    else:
        ax.plot(x, y, label=label)
    if zero_line:
        ax.hlines(y=0,
                  xmin=x[0],
                  xmax=x[-1],
                  linestyles='--',
                  label='zero',
                  colors='black',
                  linewidth=3)
    ax.legend()
    ax.set_xlabel('s')
    ax.set_ylabel('value')
    ax.set_title(title)
    st.pyplot(fig)


ss, tt = np.meshgrid(s, t, indexing='xy')
fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
surf = ax.plot_surface(ss, tt, v_tx,
                       lw=3,
                       rstride=8,
                       cstride=8,
                       alpha=0.8,
                       cmap=mpl.cm.coolwarm,
                       antialiased=False)
ax.set_xlabel('S')
ax.set_ylabel('Time to Maturity')
ax.set_title('Option Value')
st.pyplot(fig)

plot_1d(x=s,
        y=[v_tx[-1], phi],
        title='Option Value',
        label=['option', 'payoff'],
        zero_line=False
        )
if is_american:
    fig, ax = plt.subplots()
    ax.plot(t[1:],
            s_ex[1:],
            label='exercise')
    ax.hlines(y=k,
              xmin=t[0],
              xmax=t[-2],
              linestyles='--',
              label='strike',
              colors='black',
              linewidth=3)
    ax.set_title('Exercise')
    ax.set_xlabel('Time to Maturity')
    ax.set_ylabel('s')
    ax.legend()
    st.pyplot(fig)

Dt = fd.FinDiff(0, dt, 1)
Dx = fd.FinDiff(1, dx, 1)
Dxx = fd.FinDiff(1, dx, 2)

x_s = tfm.x_s(s)
x_ss = tfm.x_ss(s)

delta = Dx(v_tx) * x_s
gamma = Dxx(v_tx) * x_s**2 + Dx(v_tx) * x_ss
theta = -Dt(v_tx)

plot_1d(x=s, y=delta[-1], title='Option Delta')
plot_1d(x=s, y=gamma[-1], title='Option Gamma')
plot_1d(x=s, y=theta[-1], title='Option Theta')

st.write((v_tx**2).mean().mean()**0.5)
err = theta + c*s**2*gamma + r*s*delta - r*v_tx
st.write((err**2).mean().mean()**0.5)
fig, ax = plt.subplots()
_ = sns.heatmap(np.log10(np.abs(err)), ax=ax)
st.pyplot(fig)
fig, ax = plt.subplots()
_ = sns.heatmap(np.abs(err), ax=ax)
st.pyplot(fig)
