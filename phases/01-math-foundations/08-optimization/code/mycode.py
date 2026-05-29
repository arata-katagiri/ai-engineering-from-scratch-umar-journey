def rosenbrock(params):
    x, y = params
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2

def rosenbrock_gradient(params):
    x, y = params
    df_dx = -2 * (1 - x) + 200 * (y - x ** 2) * (-2 * x)
    df_dy = 200 * (y - x ** 2)

    return [df_dx, df_dy]



class GradientDecent:
    def __init__(self, lr=0.001):
        self.lr = lr
        self.lr_0 = self.lr
        self.steps = 0

    def step(self, params, grads):
        self.steps += 1
        return [p - self.lr_0 * 0.999**self.steps * g for p, g in zip(params, grads)]
    


class SGDMomentum:
    def __init__(self, lr=0.001, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.velocity = None

    def step(self, params, grads):
        if self.velocity is None:
            self.velocity = [0.0] * len(params)
        self.velocity = [
            self.momentum * v + g
            for v, g in zip(self.velocity, grads)
        ]
        return [p - self.lr * v for p, v in zip(params, self.velocity)]
    


class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.m = None
        self.t = 0

    def step(self, params, grads):
        if self.m is None:
            self.m = [0.0] * len(params)
            self.v = [0.0] * len(params)

        self.t += 1

        self.m = [
            self.beta1 * m + (1 - self.beta1) * g
            for m, g in zip(self.m, grads)
        ]
        self.v = [
            self.beta2 * v + (1 - self.beta2) * g ** 2
            for v, g in zip(self.v, grads)
        ]

        m_hat = [m / (1 - self.beta1 ** self.t) for m in self.m]
        v_hat = [v / (1 - self.beta2 ** self.t) for v in self.v]

        return [
            p - self.lr * mh / (vh ** 0.5 + self.epsilon)
            for p, mh, vh in zip(params, m_hat, v_hat)
        ]



def optimize(optimizer, func, grad_func, start, steps=5000):
    params = list(start)
    history = [params[:]]
    for _ in range(steps):
        grads = grad_func(params)
        params = optimizer.step(params, grads)
        history.append(params[:])
    return history

start = [-1.0, 1.0]

gd_history = optimize(GradientDecent(lr=0.0005), rosenbrock, rosenbrock_gradient, start)
sgd_history = optimize(SGDMomentum(lr=0.0001, momentum=0.9), rosenbrock, rosenbrock_gradient, start)
adam_history = optimize(Adam(lr=0.01), rosenbrock, rosenbrock_gradient, start)

for name, history in [("GD", gd_history), ("SGD+M", sgd_history), ("Adam", adam_history)]:
    final = history[-1]
    loss = rosenbrock(final)
    print(f"{name:6s} -> x={final[0]:.6f}, y={final[1]:.6f}, loss={loss:.8f}")

print ("-" * 30)

import torch

model = torch.nn.Linear(784, 10)

sgd = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
adam = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)
adamw = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(adam, T_max=100)

print ("-" * 30)

for lr in [0.0001, 0.0005, 0.001]:
    gd_history = optimize(GradientDecent(lr=lr), rosenbrock, rosenbrock_gradient, start)
    final = gd_history[-1]
    loss = rosenbrock(final)
    print(f"GD -> x={final[0]:.6f}, y={final[1]:.6f}, loss={loss:.8f}")

print ("-" * 30)


for m in [0.0, 0.5, 0.9, 0.99]:
    
    sgd_history = optimize(SGDMomentum(lr=0.0001, momentum=m), rosenbrock, rosenbrock_gradient, start)
    final = sgd_history[-1]
    loss = rosenbrock(final)
    print(f"SGD+M -> x={final[0]:.6f}, y={final[1]:.6f}, loss={loss:.8f}")

print ("-" * 30)

def func(params):
    x, y = params
    return x**2 - y**2

def func_gradient(params):
    x, y = params
    df_dx = 2 * x
    df_dy = -2 * y
    return [df_dx, df_dy]

start = [0.01, 0.01]

gd_history = optimize(GradientDecent(lr=0.0005), func, func_gradient, start)
sgd_history = optimize(SGDMomentum(lr=0.0001, momentum=0.9), func, func_gradient, start)
adam_history = optimize(Adam(lr=0.01), func, func_gradient, start)

for name, history in [("GD", gd_history), ("SGD+M", sgd_history), ("Adam", adam_history)]:
    final = history[-1]
    loss = func(final)
    print(f"{name:6s} -> x={final[0]:.6f}, y={final[1]:.6f}, loss={loss:.8f}")