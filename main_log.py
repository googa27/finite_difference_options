import streamlit as st
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pydantic as pyd
import scipy.stats as spst
import findiff as fd


import src.general as grl
import src.utils_log as ull
import config.CONFIG as CFG

with st.sidebar:
    american: bool = st.checkbox('American')
    call: bool = st.checkbox('Call')

    r = st.slider('Risk Free Rate (r)', -0.05, 0.50, 0.03)
    sig = st.slider('Volatility', -0.01, 0.90, 0.20)
    T = st.slider('Maturity', -0.01, 50.00, 5.00)
    k = st.slider('Strike Price', 0.01, 10.0, 0.40)

    nx = st.slider('nx', 1, 1000, 200)
    nt = st.slider('nt', 1, 1000, 100)

    lam = st.slider('PDE solution parameter', 0., 1., 0.5)
    log_alpha = st.slider('log_alpha', -1., -3., -2.)
    alpha = 10**log_alpha

mkt = grl.Market(r=r)
tfm = ull.TransformationLog(k=k, sig=sig, mkt=mkt)
lat = ull.Lattice(nx, nt, T, tfm, alpha=alpha)
eobs = ull.EuropeanOptionBS2(k, sig, mkt)

L = tfm.c*fd.FinDiff(0, lat.dx, 2) + tfm.r_ * \
    fd.FinDiff(0, lat.dx, 1) - mkt.r*fd.Identity()

if call:
    payoff_x = np.maximum(tfm.s(lat.x) - tfm.k, 0)
else:
    payoff_x = np.maximum(tfm.k - tfm.s(lat.x), 0)


v_tx = lat.empty_tx()
v_tx[0] = payoff_x

for i, ti in enumerate(lat.t[:-1]):
    bc = fd.BoundaryConditions((nx, ))
    # # DIRICHLET
    # bc[0] = (tfm.k*mkt.D(ti))*(1 - call)
    # bc[-1] = (tfm.s(lat.x[-1]) - mkt.D(ti)*tfm.k)*call

    # # NEUMANN
    # bc[0] = fd.FinDiff(0, lat.dx, 1), -(1 - call)
    bc[0] = fd.FinDiff(0, lat.dx, 1), 0
    bc[-1] = fd.FinDiff(0, lat.dx, 1), tfm.s(lat.x[-1])*call  # call

    # GAMMA
    bc[0] = fd.FinDiff(0, lat.dx, 2), 0
    bc[-1] = fd.FinDiff(0, lat.dx, 2), 0

    A = fd.Identity() - lat.dt*(1 - lam)*L
    B = fd.Identity() + lat.dt*lam*L
    pde = fd.PDE(A, B.matrix(lat.x.shape)@v_tx[i], bc)
    v_tx[i + 1] = pde.solve()

    if american:
        v_tx[i + 1] = np.maximum(v_tx[i + 1], payoff_x)

    # pde = fd.PDE(L, payoff_x, bc)
    # u = pde.solve()

fig, ax = plt.subplots()
ax.plot(tfm.s(lat.x), v_tx[-1])
ax.set_xlabel('x')
ax.set_ylabel('u')
ax.plot(tfm.s(lat.x), payoff_x)
st.pyplot(fig)

st.title('Analytical European Option')
fig, ax = plt.subplots()
s = tfm.s(lat.x)
if call:
    ax.plot(s, eobs.call(T, s))
    ax.plot(s, eobs.call(0, s))
else:
    ax.plot(s, eobs.put(T, s))
    ax.plot(s, eobs.put(0, s))
ax.set_xlabel('t')
ax.set_ylabel('value')
st.pyplot(fig)
