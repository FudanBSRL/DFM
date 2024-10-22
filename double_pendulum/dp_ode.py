import torch
import math
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import default_rng
from matplotlib.animation import FuncAnimation

l1 = 1
l2 = 1
m1 = 1
m2 = 1
g = 9.81

def dp(t, y):
  theta1, theta2, omega1, omega2 = y

  omega1_dot = (-g * (2 * m1 + m2) * np.sin(theta1) - m2 * g * np.sin(theta1 - 2 * theta2) - 2 * np.sin(theta1 - theta2) * m2 * (omega2**2 * l2 + omega1**2 * l1 * np.cos(theta1 - theta2))) \
                / (l1 * (2 * m1 + m2 - m2 * np.cos(2 * theta1 - 2 * theta2)))
  omega2_dot = (2 * np.sin(theta1 - theta2) * (omega1**2 * l1 * (m1 + m2) + g * (m1 + m2) * np.cos(theta1) + omega2**2 * l2 * np.cos(theta1 - theta2))) \
                / (l2 * (2 * m1 + m2 - m2 * np.cos(2 * theta1 - 2 * theta2)))
  
  return [omega1, omega2, omega1_dot, omega2_dot]


def dp_ode(time_span, ic, t_eval, method):
  dp_soln = solve_ivp(dp, time_span, ic, method=method, t_eval=t_eval, rtol=1e-10, atol=1e-10)
  return dp_soln


if __name__ == '__main__':
  maxtime = 40
  t_eval = np.linspace(0, maxtime, int(40 * maxtime + 1))
  sol = dp_ode(time_span=[0, maxtime], ic=[math.pi * 1 / 4 , math.pi * 1 / 5, 0, 0], t_eval=t_eval, method="RK45")
  theta = sol.y
  
  theta1, theta2 = sol.y[0], sol.y[1]
  x1 = l1 * np.sin(theta1)
  y1 = -l1 * np.cos(theta1)
  x2 = x1 + l2 * np.sin(theta2)
  y2 = y1 - l2 * np.cos(theta2)

  fig, ax = plt.subplots()
  # ax.axis('equal')
  # fig.set_size_inches(5, 5)
  ax.set_aspect(1)
  ax.set_xlim(-2.5, 2.5)
  ax.set_ylim(-2.5, 2.5)
  line, = ax.plot([], [], 'o-', lw=2)

  def init():
    line.set_data([], [])
    return line,

  def animate(i):
      x = [0, x1[i], x2[i]]
      y = [0, y1[i], y2[i]]
      line.set_data(x, y)
      return line,

  ani = FuncAnimation(fig, animate, frames=len(t_eval), interval=(t_eval[1] - t_eval[0]) * 1000, blit=True, init_func=init)
  ani.save("test.gif")
  # fig.savefig("test.jpg")
  plt.show()
  print(sol)

  fig2, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, ncols=1, sharex=False)
  ax1.plot(sol.t, sol.y[0, :], linestyle='-')
  ax2.plot(sol.t, sol.y[1, :], linestyle='-')
  ax3.plot(sol.t, sol.y[2, :], linestyle='-')
  ax4.plot(sol.t, sol.y[3, :], linestyle='-')
  fig2.savefig("test.png", dpi=500)
  
  
