import numpy as np
from lqgame import LQGame

class Unicycle:
    def __init__(self, T, N, x0, Q, R, QT, l, lT, r):
        self.T = T
        self.N = N
        self.x0 = x0.copy()
        self.x = x0.copy()
        self.dt = 0.1
        self.reset()
        self.xdim, self.udim = N*4, 2
        self.Q, self.R, self.QT, self.l, self.lT, self.r = Q, R, QT, l, lT, r

    def reset(self):
        self.t = 0
        self.x = self.x0
        self.reward = np.zeros(self.N)
        return self.x, False

    def step(self, u):
        # Accumulate reward collected on this step
        for i in range(self.N):
            cost_state = 1/2 * np.dot(self.x, self.Q[i] @ self.x) + np.dot(self.l[self.t, i], self.x)
            cost_control = 1/2 * np.sum(
                [np.dot(u[j], self.R[i,j] @ u[j]) + 2*np.dot(self.r[self.t,i,j], u[j]) \
                        for j in range(self.N)])
            self.reward[i] += cost_state + cost_control
        # Step the system forward
        change = np.zeros(self.x0.shape)
        for i in range(self.N):
            ai, wi = u[i]
            vi = self.x[i*4+2]
            thetai = self.x[i*4+3]
            change[i*4] = vi * np.cos(thetai)
            change[i*4+1] = vi * np.sin(thetai)
            change[i*4+2] = ai
            change[i*4+3] = wi
        self.x = self.x + self.dt * change
        # Increase step counter
        self.t += 1
        # Add in the terminal cost
        if self.t == self.T:
            for i in range(self.N):
                self.reward[i] += 1/2 * np.dot(self.x, self.QT[i] @ self.x) + \
                        np.dot(self.lT[i], self.x)
        return self.x, (self.t < self.T)

    def get_traj(self, cont):
        ''' Returns a state and a control trajectory with the given controller '''
        self.reset()
        cont.reset()
        xtraj = [self.x0]
        utraj = []
        for t in range(self.T):
            u = cont.control(self.x)
            x, _ = self.step(u)
            xtraj.append(x)
            utraj.append(u)
        xtraj = np.array(xtraj)
        utraj = np.array(utraj)
        return xtraj, utraj

    def linearize_traj(self, xtraj, utraj):
        ''' Returns an LQGame resulted from linearizing the dynamics around
        given traj'''
        N = self.N
        A = np.zeros((self.T, self.xdim, self.xdim))
        B = np.zeros((self.T, N, self.xdim, self.udim))
        l = np.zeros((self.T, N, self.xdim))
        lT = np.zeros((N, self.xdim))
        r = np.zeros((self.T, N, N, self.udim))
        for i in range(N):
            lT[i] = np.dot(self.Q[i], xtraj[-1]) + self.lT[i]
        for t in range(self.T):
            A[t] = np.eye(self.xdim)
            for i in range(N):
                vi = xtraj[t, i*4+2]
                thetai = xtraj[t, i*4+3]
                ai, wi = utraj[t,i]
                xselfind = np.arange(i*4, i*4+4)
                A[t,i*4:i*4+2, i*4+2:i*4+4] = np.array([
                    [self.dt * np.cos(thetai), -self.dt * vi * np.sin(thetai)],
                    [self.dt * np.sin(thetai), self.dt * vi * np.cos(thetai)]
                    ])
                B[t,i, xselfind] = np.array([
                    [0, 0],
                    [0, 0],
                    [1, 0],
                    [0, 1]]) * self.dt
                l[t, i] = np.dot(self.Q[i], xtraj[t]) + self.l[t,i]
                for j in range(N):
                    r[t, i, j] = np.dot(self.R[i,j], utraj[t, j]) + self.r[t,i,j]

        x0 = np.zeros(self.x0.shape)
        return LQGame(self.N, A, B, self.Q, self.R, self.QT, self.T, l, lT, r, x0)
