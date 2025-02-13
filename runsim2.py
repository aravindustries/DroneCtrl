import numpy as np
import matplotlib.pyplot as plt

class Integrator:
    """Integrator for a system of first-order ordinary differential equations
    of the form dot x = f(t, x, u).
    """
    def __init__(self, dt, f):
        self.dt = dt
        self.f = f

    def step(self, t, x, u):
        raise NotImplementedError

class Euler(Integrator):
    def step(self, t, x, u):
        return x + self.dt * self.f(t, x, u)

class Heun(Integrator):
    def step(self, t, x, u):
        intg = Euler(self.dt, self.f)
        xe = intg.step(t, x, u)
        return x + 0.5*self.dt * (self.f(t, x, u) + self.f(t+self.dt, xe, u))

class RungeKutta4(Integrator):
    def step(self, t, x, u):
        X1 = self.f(t, x, u)
        X2 = self.f(t + self.dt/2, x + self.dt/2 * X1, u)
        X3 = self.f(t + self.dt/2, x + self.dt/2 * X2, u)
        X4 = self.f(t + self.dt, x + self.dt * X3, u)
        return x + self.dt/6 * (X1 + 2*X2 + 2*X3 + X4)

def equations_of_motion(t, state, u):
    mass, Jx, Jy, Jz, Mx, My, Mz, fx, fy, fz = u
    pn, pe, pd, u, v, w, phi, theta, psi, p, q, r = state
    
    # Rotation matrix from body to inertial frame
    c_phi, s_phi = np.cos(phi), np.sin(phi)
    c_theta, s_theta = np.cos(theta), np.sin(theta)
    c_psi, s_psi = np.cos(psi), np.sin(psi)
    
    R_bi = np.array([
        [c_theta * c_psi, s_phi * s_theta * c_psi - c_phi * s_psi, c_phi * s_theta * c_psi + s_phi * s_psi],
        [c_theta * s_psi, s_phi * s_theta * s_psi + c_phi * c_psi, c_phi * s_theta * s_psi - s_phi * c_psi],
        [-s_theta, s_phi * c_theta, c_phi * c_theta]
    ])
    
    # Translational equations
    vel_inertial = R_bi @ np.array([u, v, w])
    udot = r * v - q * w + fx / mass
    vdot = p * w - r * u + fy / mass
    wdot = q * u - p * v + fz / mass
    
    # Angular rate kinematics
    T_matrix = np.array([
        [1, s_phi * np.tan(theta), c_phi * np.tan(theta)],
        [0, c_phi, -s_phi],
        [0, s_phi / c_theta, c_phi / c_theta]
    ])
    ang_rates = T_matrix @ np.array([p, q, r])
    
    # Moments of inertia terms
    Gamma1 = (Jz - Jy) / Jx
    Gamma2 = (Jz - Jx) / Jy
    Gamma3 = (Jy - Jx) / Jz
    Gamma4 = 1 / Jx
    Gamma5 = 1 / Jy
    Gamma6 = 1 / Jz
    
    # Angular equations
    pdot = Gamma1 * q * r + Gamma4 * Mx
    qdot = Gamma2 * p * r + Gamma5 * My
    rdot = Gamma3 * p * q + Gamma6 * Mz
    
    return np.array([*vel_inertial, udot, vdot, wdot, *ang_rates, pdot, qdot, rdot])

def simulate_drone(dt, T, initial_state, mass, Jx, Jy, Jz, Mx, My, Mz, fx, fy, fz, method="RK4"):

    integrators = {"Euler": Euler, "Heun": Heun, "RK4": RungeKutta4}
    integrator = integrators[method](dt, equations_of_motion)
    
    state = np.array(initial_state)
    t_values = np.arange(0, T + dt, dt) 
    states = [state]
    
    for t in t_values[1:]:
        state = integrator.step(t, state, (mass, Jx, Jy, Jz, Mx, My, Mz, fx, fy, fz))
        states.append(state)
    
    return t_values, np.array(states)

def main():
    dt = 0.01
    T = 10
    initial_state = np.zeros(12)
    mass, Jx, Jy, Jz = 1.0, 0.1, 0.1, 0.2
    Mx, My, Mz = 0, 0, 1
    fx, fy, fz = 0, 0, -9.81 * mass
    
    t_values, states = simulate_drone(dt, T, initial_state, mass, Jx, Jy, Jz, Mx, My, Mz, fx, fy, fz)
    
    
    state_labels = ['pn', 'pe', 'pd', 'u', 'v', 'w', 'phi', 'theta', 'psi', 'p', 'q', 'r']

    for i in range(states.shape[1]):
        plt.plot(t_values, states[:, i], label=state_labels[i])
    
    plt.xlabel("Time (s)")
    plt.ylabel("State Value")
    plt.title("Drone States Over Time")
    plt.legend()
    plt.savefig('sim.png')

if __name__ == "__main__":
    main()
