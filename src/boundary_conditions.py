import findiff as fd
import numpy as np
import pydantic as pyd


class BcHandler():
    pass


def _set_dirichlet(bc,
                   option_payoff: str,
                   s: np.ndarray):
    is_call: bool = option_payoff == 'call'
    bc[0] = mkt.D(th)*k*(not is_call)
    bc[1] = (s[-1] - mkt.D(th)*k)*is_call


def _set_neumann(bc, option_payoff: str):
    is_call: bool = option_payoff == 'call'
    bc[0] = fd.FinDiff(0, ds, 1), -1*(not is_call)
    bc[1] = fd.FinDiff(0, ds, 1), 1*is_call


def _set_gamma(bc):
    bc[0] = fd.FinDiff(0, ds, 2), 0
    bc[1] = fd.FinDiff(0, ds, 2), 0


def set_bc(bc,
           boundary_contidion: str,
           option_payoff: str) -> None:
    if boundary_contidion == 'dirichlet':
        _set_dirichlet()
    elif boundary_contidion == 'neumann':
        _set_neumann()
    elif boundary_contidion == 'gamma':
        _set_gamma()
    else:
        raise ValueError(f'Wrong bc string: {boundary_contidion}')
