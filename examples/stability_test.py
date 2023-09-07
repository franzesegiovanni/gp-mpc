import torch

from mpc import mpc
from mpc.mpc import QuadCost, LinDx, GradMethods
from mpc.env_dx import cartpole

import numpy as np

import matplotlib.pyplot as plt

from tqdm import tqdm



dx = cartpole.CarDx()

mpc_T = 25 # steps of MPC

torch.manual_seed(0)
th = uniform(n_batch, -2*np.pi, 2*np.pi)
thdot = uniform(n_batch, -.5, .5)
x = uniform(n_batch, -0.5, 0.5)
xdot = uniform(n_batch, -0.5, 0.5)
xinit = torch.stack((x, xdot, torch.cos(th), torch.sin(th), thdot), dim=1)

#inizialize with the value of the 
x = xinit
u_init = torch.zeros(1, 1, dx.n_ctrl)

q, p = dx.get_true_obj()

Q = torch.diag(q).unsqueeze(0).unsqueeze(0).repeat(
    mpc_T, 1, 1, 1
)
p = p.unsqueeze(0).repeat(mpc_T, 1, 1)

action_history = []

nominal_states, nominal_actions, nominal_objs = mpc.MPC(
    dx.n_state, dx.n_ctrl, mpc_T,
    u_init=u_init,
    u_lower=dx.lower, u_upper=dx.upper,
    lqr_iter=50,
    verbose=0,
    exit_unconverged=False,
    detach_unconverged=False,
    linesearch_decay=dx.linesearch_decay,
    max_linesearch_iter=dx.max_linesearch_iter,
    grad_method=GradMethods.AUTO_DIFF,
    eps=1e-2,
)(x, QuadCost(Q, p), dx)
    
next_action = nominal_actions[0]
u_init = torch.cat((nominal_actions[1:], torch.zeros(1, 1, dx.n_ctrl)), dim=0)
# u_init[-2] = u_init[-3]
x = dx(x, next_action)