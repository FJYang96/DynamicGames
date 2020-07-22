import numpy as np
import scipy.linalg as sla

class LQGame:
    def __init__(self, N, A, B, Q, R, QT, T, l, lT, r,
                 x0=None,
                 noise_var=0):
        self.N = N
        self.A, self.B, self.Q, self.R, self.QT = A, B, Q, R, QT
        self.xdim, self.udim = B.shape[2:]
        self.T = T
        self.l, self.r, self.lT = l, r, lT
        self.noise_var = noise_var
        if x0 is None:
            self.x0 = (np.random.random(self.xdim) - 0.5) * 5
        else:
            self.x0 = x0
        self.solution_computed = False
        self.reset()

    def reset(self):
        self.x = self.x0
        self.t = 0
        self.reward = np.zeros(self.N)
        return self.x, False

    def get_params(self):
        return self.N, self.A[self.t:], self.B[self.t:], self.Q, self.R, self.QT, \
            self.T, self.l[self.t:], self.lT, self.x0, self.noise_var

    def step(self, u):
        ''' Returns a (vector, bool) tuple
        Steps the system forward for one step
        '''
        # Accumulate the reward collected on this step
        for i in range(self.N):
            cost_state = 1/2 * np.dot(self.x, self.Q[i] @ self.x) + np.dot(self.l[self.t, i], self.x)
            cost_control = 1/2 * np.sum(
                [np.dot(u[j], self.R[i,j] @ u[j]) + 2*np.dot(self.r[self.t,i,j], u[j]) \
                        for j in range(self.N)])
            self.reward[i] += cost_state + cost_control
        # Compute the next state
        self.x = self.A[self.t] @ self.x
        self.x = self.x + np.sum([self.B[self.t,i] @ u[i] for i in range(self.N)], axis=0)
        self.x = self.x + np.random.normal(0, self.noise_var, len(self.x))
        # Increase step counter
        self.t += 1
        # Add in the terminal cost
        if self.t == self.T:
            for i in range(self.N):
                self.reward[i] += 1/2 * np.dot(self.x, self.QT[i] @ self.x) + \
                        np.dot(self.lT[i], self.x)
        return (self.x, self.t >= self.T)

    def RBZB_inv(self, Z, t):
        ''' Return a block matrix
        Compute and store the RBZB matrix used in the recursive steps
        '''
        Rii = [self.R[i,i,:,:] for i in range(self.N)]
        R = sla.block_diag(*Rii)
        BZ = np.block([[self.B[t, i].T @ Z[i]] for i in range(self.N)])
        Bi = np.block([self.B[t, i] for i in range(self.N)])
        RBZB = R + BZ @ Bi
        return np.linalg.inv(RBZB)

    def recursive_step(self, Z, xi, t):
        # Find P, alfa
        RBZBinv = self.RBZB_inv(Z, t)
        BZA = np.block([[self.B[t, i].T @ Z[i] @ self.A[t]] \
                for i in range(self.N)])
        Bxi = np.block([[self.B[t, i].T @ xi[i,:,None] + \
                self.r[t,i,i,:,None]] for i in range(self.N)])
        '''
        print(self.B[t, 0].T @ xi[0,:,None])
        print(self.r[t,0,0,:,None])
        print(Bxi)
        print('-'*30)
        '''
        P = (RBZBinv @ BZA).reshape(self.N, self.udim, self.xdim)
        alfa = (RBZBinv @ Bxi).reshape(self.N, self.udim)
        # Find F
        F = self.A[t] - np.sum([self.B[t, i] @ P[i] for i in range(self.N)], axis=0)
        # Find Z, xi
        Z_new = np.zeros((self.N,self.xdim,self.xdim))
        xi_new = np.zeros((self.N,self.xdim))
        beta = -np.sum([self.B[t, i] @ alfa[i][:,None] \
                for i in range(self.N)], axis=0)
        for i in range(self.N):
            Z_new[i] = F.T @ Z[i] @ F + \
                        np.sum([P[j].T @ self.R[i,j] @ P[j] for j in range(self.N)], axis=0) + \
                        self.Q[i]
            xi_new_i = self.l[t, i][:, None] + F.T @ (xi[i][:, None] + Z[i] @ beta) + np.sum(
                    [P[j].T @ (self.R[i,j] @ alfa[j][:, None] - \
                        self.r[t,i,j][:, None]) for j in range(self.N)], axis=0)
            xi_new[i] = xi_new_i[:, 0]
        return P, alfa, Z_new, xi_new

    def find_sigma(self):
        # Compute the control covariance matrix
        for t in range(self.T-1):
            for i in range(self.N):
                self.sigma[t,i] = np.linalg.inv(
                        self.R[i,i] + \
                                self.B[t, i].T @ self.Z[t+1,i] @ self.B[t, i])
        for i in range(self.N):
            self.sigma[self.T-1, i] = np.linalg.inv(
                    self.R[i,i] + \
                            self.B[self.T-1, i].T @ self.QT[i] @ self.B[self.T-1, i])

    def solve(self):
        self.Z = np.zeros((self.T, self.N, self.xdim, self.xdim))
        self.xi = np.zeros((self.T, self.N, self.xdim))
        self.P = np.zeros((self.T, self.N, self.udim, self.xdim))
        self.alfa = np.zeros((self.T, self.N, self.udim))
        self.sigma = np.zeros((self.T, self.N, self.udim, self.udim))
        # Recursive Step
        P, alfa, Z, xi = self.recursive_step(self.QT, self.lT, self.T-1)
        T = self.T
        self.P[T-1], self.alfa[T-1], self.Z[T-1], self.xi[T-1] = P, alfa, Z, xi
        for t in range(self.T-2,-1,-1):
            P, alfa, Z, xi = self.recursive_step(Z, xi, t)
            self.P[t], self.alfa[t], self.Z[t], self.xi[t] = P, alfa, Z, xi
        self.find_sigma()
        self.solution_computed = True
        return self.P, self.alfa, self.sigma

    def generate_trajectory(self, controller=None):
        if controller is None:
            P, alfa, sigma = self.solve()
            controller = LQGcontrol(self.N, self.T, P, alfa, sigma)
        x, _ = self.reset()
        xtraj = [x]
        utraj = []
        for i in range(self.T):
            u = controller.control(x)
            x, done = self.step(u)
            xtraj.append(x)
            utraj.append(u)
        xtraj = np.array(xtraj)
        utraj = np.array(utraj)
        return xtraj, utraj

    def linearize_traj(self, xtraj, utraj):
        l = np.zeros(self.l.shape)
        lT = np.zeros(self.lT.shape)
        r = np.zeros(self.r.shape)
        for t in range(self.T):
            for i in range(self.N):
                l[t, i] = np.dot(self.Q[i], xtraj[t]) + self.l[t,i]
                for j in range(self.N):
                    r[t,i,j] = np.dot(self.R[i,j], utraj[t,j]) + self.r[t,i,j]
        for i in range(self.N):
            lT[i] = np.dot(self.QT[i], xtraj[-1]) + self.lT[i]
        x0 = np.zeros(self.x.shape)
        return LQGame(self.N, self.A, self.B, self.Q, self.R, self.QT,
                self.T, l, lT, r, x0)

class LQGcontrol:
    def __init__(self, N, T, P, alfa, sigma,
                 stochastic=False):
        self.N = N
        self.udim, self.xdim = P.shape[2:]
        self.P, self.alfa, self.sigma = P, alfa, sigma
        self.stochastic = stochastic
        self.reset()

    def reset(self):
        self.t = 0

    def control(self, x):
        u = np.zeros((self.N, self.udim))
        for i in range(self.N):
            u[i] = -self.P[self.t,i] @ x - self.alfa[self.t,i]
            if self.stochastic:
                u[i] = np.random.multivariate_normal(u[i], self.sigma[self.t, i])
        self.t += 1
        return u

class MPCLQControl(LQGcontrol):
    def __init__(self, N, A, B, Q, R, QT, T, l, lT, x0, noise_var,
                 stochastic=False):
        self.N = N
        self.A, self.B, self.Q, self.R, self.QT = A, B, Q, R, QT
        self.xdim, self.udim = B.shape[2:]
        self.T = T
        self.l, self.lT = l, lT
        self.noise_var = noise_var
        self.x0 = x0
        self.stochastic = stochastic
        self.reset()

    def reset(self):
        self.t = 0

    def control(self, x):
        game = LQGame(self.N, self.A, self.B, self.Q, self.R, self.QT,
                      self.T-self.t, self.l, self.lT, self.r, x0=x, noise_var=0)
        P, alfa, sigma = game.solve()
        u = np.zeros((self.N, self.udim))
        for i in range(self.N):
            u[i] = -P[0,i] @ x - alfa[0,i]
            if self.stochastic:
                u[i] = np.random.multivariate_normal(u[i], sigma[0, i])
        self.t += 1
        return u
