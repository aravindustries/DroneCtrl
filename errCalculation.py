import numpy as np
import matplotlib.pyplot as plt
import integrators as intg

# Mass-spring system parameters
m = 1
b = 0.25
k = 1

# Nonlinear state space form:
#  xdot = f(t, x, u), 
#   t: time
#   x: state vector
#   u: input
def f(t, x, u):
    return np.array([x[1], -b/m * x[1] - k/m * x[0] + 1/m * u])

# Initial conditions
t = 0
x = np.array([1, 0])
u = 0
dt = 0.1
n = 500

euler_integrator = intg.Euler(dt, f)
heun_integrator = intg.Heun(dt, f)
runge_kutta_integrator = intg.RungeKutta4(dt, f)

t_history_euler = [t]
x_history_euler = [x]
for i in range(n):
    x = euler_integrator.step(t, x, u)
    t = (i+1) * dt
    t_history_euler.append(t)
    x_history_euler.append(x)

t = 0
x = np.array([1, 0])

t_history_heun = [t]
x_history_heun = [x]
for i in range(n):
    x = heun_integrator.step(t, x, u)
    t = (i+1) * dt
    t_history_heun.append(t)
    x_history_heun.append(x)

t = 0
x = np.array([1, 0])

t_history_runge_kutta = [t]
x_history_runge_kutta = [x]
for i in range(n):
    x = runge_kutta_integrator.step(t, x, u)
    t = (i+1) * dt
    t_history_runge_kutta.append(t)
    x_history_runge_kutta.append(x)

x_history_euler = np.array(x_history_euler)
x_history_heun = np.array(x_history_heun)
x_history_runge_kutta = np.array(x_history_runge_kutta)

def analytical_solution(t):
    omega = np.sqrt(k/m - (b/(2*m))**2)
    return np.exp(-b/(2*m)*t) * (np.cos(omega*t) + (b/(2*m*omega))*np.sin(omega*t))

t_points = np.array(t_history_euler)
x_analytical = analytical_solution(t_points)

euler_error = np.mean(np.abs(x_history_euler[:, 0] - x_analytical))
heun_error = np.mean(np.abs(x_history_heun[:, 0] - x_analytical))
runge_kutta_error = np.mean(np.abs(x_history_runge_kutta[:, 0] - x_analytical))

t_analytical = np.linspace(0, n*dt, 1000)
x_analytical_smooth = analytical_solution(t_analytical)

plt.figure()
plt.plot(t_history_euler, x_history_euler[:, 0], label='Euler Method')
plt.plot(t_history_heun, x_history_heun[:, 0], label='Heun Method')
plt.plot(t_history_runge_kutta, x_history_runge_kutta[:, 0], label='Runge-Kutta Method')
plt.plot(t_analytical, x_analytical_smooth, label='Analytical Solution', linestyle='dashed')
plt.xlabel('Time')
plt.ylabel('Position')
plt.legend()
plt.title('Mass-Spring System Simulation')
plt.savefig('frnco.png')

print("\nAverage Absolute Errors:")
print(f"Euler Method: {euler_error:.6f}")
print(f"Heun Method: {heun_error:.6f}")
print(f"Runge-Kutta Method: {runge_kutta_error:.6f}")
