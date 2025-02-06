class Integrator:
    """Integrator for a system of first-order ordinary differential equations
    of the form \dot x = f(t, x, u).
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
        xe = intg.step(t, x, u) # Euler predictor step
        return x + 0.5*self.dt * (self.f(t, x, u) + self.f(t+self.dt, xe, u))

class RungeKutta4(Integrator):
    def step(self, t, x, u):
        X1 = self.f(t, x, u)
        X2 = self.f(t + self.dt/2, x + self.dt/2 * X1, u)
        X3 = self.f(t + self.dt/2, x + self.dt/2 * X2, u)
        X4 = self.f(t + self.dt, x + self.dt * X3, u)
        return x + self.dt/6 * (X1 + 2*X2 + 2*X3 + X4)
