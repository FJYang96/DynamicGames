import numpy as np
import matplotlib.pyplot as plt
from lqgame import LQGame, LQGcontrol

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']

def rand_pd_matrix(dim):
    a = np.random.random((dim,dim)) * 5
    return a.T @ a + np.eye(dim)

def double_int_2D(N, T, coupled=False):
    xdim = 4*N
    A = np.tile(np.eye(xdim), (T,1,1))
    B = np.zeros((T, N, xdim, 2))
    ux_ind = tuple([i*4+1 for i in range(N)])
    uy_ind = tuple([i*4+3 for i in range(N)])
    for t in range(T):
        for i in range(N):
            A[t, i*4, i*4+1] = 1
            A[t, i*4+2, i*4+3] = 1
            if coupled:
                B[t, i, ux_ind, 0] = 0.05
                B[t, i, uy_ind, 1] = 0.05
            # Overwrite the control authority on oneself
            B[t, i, i*4+1, 0] = 0.1
            B[t, i, i*4+3, 1] = 0.1
    return A, B

def plot_double_int_2D_traj(traj, N, randsample=False):
    for i in range(N):
        if randsample:
            plt.plot(traj[:,i*4], traj[:,i*4+2], color=colors[i], alpha=0.4)
        else:
            plt.plot(traj[:,i*4], traj[:,i*4+2], '*-', color=colors[i])

def plot_unicycle_traj(traj, N, randsample=False):
    if randsample:
        for i in range(N):
            plt.plot(traj[:, i*4], traj[:, i*4+1], color=colors[i], alpha=0.4)
    else:
        for i in range(N):
            plt.plot(traj[:, i*4], traj[:, i*4+1], '*-', color=colors[i])

def generate_traj(env, controller):
    N, T = env.N, env.T
    controller.reset()
    x, _ = env.reset()
    traj = [x]
    for i in range(T):
        u = controller.control(x)
        x, done = env.step(u)
        traj.append(x)
    traj = np.array(traj)
    return traj

def random_double_int_env(N, T):
    xdim = 4*N
    udim = 2
    A, B = double_int_2D(N, T)
    Q = np.zeros((N, xdim, xdim))
    R = np.zeros((N,N,udim,udim))
    QT = np.zeros((N, xdim, xdim))
    l = np.zeros((T, N, xdim))
    lT = np.zeros((N, xdim))
    r = np.zeros((T, N, N, udim))
    for i in range(N):
        Q[i] = rand_pd_matrix(xdim)
        QT[i] = rand_pd_matrix(xdim)
        for j in range(N):
            R[i,j] = rand_pd_matrix(udim)
    return LQGame(N, A, B, Q, R, QT, T, l, lT, r)


class TrajControl:
    def __init__(self, utraj):
        self.utraj = utraj
        self.t = 0
    def reset(self):
        self.t = 0
    def control(self, x):
        c = self.utraj[self.t]
        self.t += 1
        return c

class RandControl():
    def __init__(self, N, udim, mult=1):
        self.t = 0
        self.N = N
        self.udim = udim
        self.mult = mult
    def reset(self):
        self.t = 0
    def control(self, x):
        return (np.random.random((self.N, self.udim)) - 1/2) * self.mult

class ZeroControl(RandControl):
    def control(self, x):
        return np.zeros((N, udim))

def improve_traj_once(g, P_delta, alfa_delta, sigma, xtraj, utraj, multiplier=1):
    cont = LQGcontrol(g.N, g.T, P_delta, multiplier * alfa_delta, sigma)
    x, _ = g.reset()
    new_xtraj = [x]
    new_utraj = []
    for i in range(g.T):
        delta_xi = x - xtraj[i]
        delta_u = cont.control(delta_xi)
        u = utraj[i] + delta_u
        x, _ = g.step(u)
        new_xtraj.append(x)
        new_utraj.append(u)
    new_xtraj = np.array(new_xtraj)
    new_utraj = np.array(new_utraj)
    cont.reset()
    return new_xtraj, new_utraj, cont

def improve_traj(g, P_delta, alfa_delta, sigma, xtraj, utraj, tol=1e-1):
    multiplier = 1
    new_xtraj, new_utraj, cont = improve_traj_once(
            g, P_delta, alfa_delta, sigma, xtraj, utraj)
    while(np.max(np.abs(new_xtraj.ravel() - xtraj.ravel()) > tol)):
        multiplier /= 2
        new_xtraj, new_utraj, cont = improve_traj_once(
                g, P_delta, alfa_delta, sigma, xtraj, utraj, multiplier=multiplier)
    return new_xtraj, new_utraj, cont

def linearize_and_improve(g, xtraj, utraj, tol=1e-1):
    lqapprx = g.linearize_traj(xtraj, utraj)
    P, alfa, sigma = lqapprx.solve()
    new_xtraj, new_utraj, cont = improve_traj(g, P, alfa, sigma, xtraj, utraj, tol)
    return new_xtraj, new_utraj, cont

def ilqgame(g, xtraj, utraj, improve_tol=1e-1, ilq_tol=1e-2):
    counter = 0
    new_xtraj, new_utraj, cont = linearize_and_improve(g, xtraj, utraj, improve_tol)
    while(np.max(np.abs(new_xtraj.ravel() - xtraj.ravel()) > ilq_tol)
            and counter < 100):
        xtraj, utraj = new_xtraj.copy(), new_utraj.copy()
        new_xtraj, new_utraj, cont = linearize_and_improve(g, xtraj, utraj, improve_tol)
        counter += 1
    return new_xtraj, new_utraj, cont

def generate_nonlinear_traj(g, sol_x, sol_u, cont, stochastic=False):
    cont.stochastic = stochastic
    cont.reset()
    x, _ = g.reset()
    xtraj = [x]
    utraj = []
    for t in range(g.T):
        delta_xt = x - sol_x[t]
        delta_u = cont.control(delta_xt)
        u = sol_u[t] + delta_u
        x, _ = g.step(u)
        xtraj.append(x)
        utraj.append(u)
    xtraj = np.array(xtraj)
    utraj = np.array(utraj)
    return xtraj, utraj
