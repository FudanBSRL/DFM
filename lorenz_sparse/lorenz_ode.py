import torch
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import numpy as np

sigma = 10
beta = 8 / 3
rho = 28

def lorenz(t, y):
  
  dy0 = sigma * (y[1] - y[0])
  dy1 = y[0] * (rho - y[2]) - y[1]
  dy2 = y[0] * y[1] - beta * y[2]
  
  # since lorenz is 3-dimensional, dy/dt should be an array of 3 values
  return [dy0, dy1, dy2]


def lorenz_ode(time_span, ic, t_eval, method):
  lorenz_soln = solve_ivp(lorenz, time_span, ic, method=method, t_eval=t_eval, rtol=1e-10, atol=1e-10)
  # lorenz_soln = solve_ivp(lorenz, time_span, ic, method=method)
  # lorenz_soln = solve_ivp(lorenz, time_span, ic , t_eval=t_eval, method=method)
  return lorenz_soln
  
