import streamlit as st
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import src.utils as ul
import config.CONFIG as CFG

st.write('We define the following notations:')

eqs_1: str = r'''
\begin{align}
    \theta &:= T - t\\
    \gamma &:= \frac{\sigma^2}{2}\\
    \mu &:= r - \gamma\\
\end{align}
'''

eqs_2: str = r'''
\begin{align}
    \frac{dS}{S} &= r \, dt + \sigma \, dW\\
    x &= \log\left(\frac{S}{K}\right) + \mu \, \theta\\
    S &= Ke^{x - \mu \theta}\\
    dx &= \sigma \, dW
\end{align}
'''
eqs_3: str = r'''
\begin{align}
    \mathcal{L}v &:= -v_{\theta} + \sigma^2 / 2 v_{xx} - rv\\
    \frac{\mathcal L V}{V_x} &= \frac{\mathcal L S}{S_x}\\
    \mathcal{L} V &= 0\\
    -V_{\theta} + \sigma^2 / 2 V_{xx} - rV &= 0\\
    V - \phi &\geq 0\\
    V(0) &= \phi\\
    V &= Du\\
    D_{\theta} &= -r\theta\\
    V_{\theta} &= D[-ru + u_{\theta}]\\
    -u_{\theta} + \sigma^2 / 2 u_{xx} &= 0
\end{align}
'''

st.latex(eqs_1)
st.write('We have, under the risk-neutral measure:')
st.latex(eqs_2)
st.write('We have the following equations:')
st.latex(eqs_3)
st.write('Remark that $x$ is quite similar to the upper half of $d_{-}$')

with st.sidebar:
    american: bool = st.checkbox('American')
    call: bool = st.checkbox('Call')

    r = st.slider('Risk Free Rate (r)', -0.05, 0.50, 0.03)
    sig = st.slider('Volatility', -0.01, 0.90, 0.20)
    T = st.slider('Maturity', -0.01, 50.00, 5.00)
    k = st.slider('Strike Price', 0.01, 10.0, 0.40)

    nx = st.slider('nx', 1, 1000, 200)
    nt = st.slider('nt', 1, 1000, 100)

sig_x = sig*T**0.5
r_adj = (r - sig**2/2)
z_a = 3  # 1.96

mkt = ul.Market(r=r)
# lat = ul.Lattice(nx, nt, T, (r_adj - 4*sig_x, -r_adj + sig_x))
# lat = ul.Lattice(nx, nt, T, (-2*z_a*sig_x, z_a*sig_x))
s_lower, s_upper = CFG.EPS, 2*k
x_lower, x_upper = np.log(s_lower/k) + r_adj*T, np.log(s_upper/k) + r_adj*T
x_upper = max(x_upper, z_a*sig_x)
lat = ul.Lattice(nx, nt, T, (x_lower, x_upper))
fdo = ul.FinDiffOption(lat, sig, mkt, k)
eobs = ul.EuropeanOptionBS(k, sig, mkt)


s = fdo.tfm.s(0, lat.x)
# fig, ax = plt.subplots()
# ax.plot(s)
# st.pyplot(fig)
v_tx, s_ex = fdo.solve_heat(call=call, american=american)

# st.title('Price')
tt, xx = lat.meshgrid_tx()
ss = fdo.tfm.s(tt, xx)
fig, ax = plt.subplots()
ctrf = ax.contourf(tt, ss, v_tx, cmap=mpl.cm.coolwarm, levels=100)

if american and not call:
    ax.plot(lat.t[:-1], s_ex[:-1], color='black',
            linewidth=3, label='Exercise')

ax.legend()
ax.set_xlabel('t')
ax.set_ylabel('s')

fig.colorbar(ctrf)

# ax.matshow(v_tx)
st.pyplot(fig)


st.title('Final Value (Numerical)')
fig, ax = plt.subplots()
ax.plot(s, v_tx[0], label='Option')
ax.plot(s, v_tx[-1], label='Payoff')
ax.set_xlabel('s')
ax.set_ylabel('value')
ax.legend()
st.pyplot(fig)

if american:
    st.title('Exercise')
    fig, ax = plt.subplots()
    ax.plot(lat.t[:-5], s_ex[:-5], color='black',
            linewidth=3, label='Exercise')
    ax.set_xlabel('t')
    ax.set_ylabel('s')
    ax.legend()
    st.pyplot(fig)
# st.write(eobs.put(T, s))


st.title('Analytical European Option')
fig, ax = plt.subplots()
if call:
    ax.plot(s, eobs.call(T, s))
    ax.plot(s, eobs.call(0, s))
else:
    ax.plot(s, eobs.put(T, s))
    ax.plot(s, eobs.put(0, s))
ax.set_xlabel('t')
ax.set_ylabel('value')
st.pyplot(fig)
