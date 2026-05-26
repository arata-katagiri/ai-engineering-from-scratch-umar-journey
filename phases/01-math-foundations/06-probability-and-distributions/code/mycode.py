# 1

import math
import random

def factorial(n):
    result = 1
    for i in range(2, n + 1):
        result *= i

    return result

def combinations(n, k):
    return factorial(n) // (factorial(k) * factorial(n - k))

def conditional_probability(p_a_and_b, p_b):
    return p_a_and_b / p_b

p_king_given_face = conditional_probability(4/52, 12/52)
print(f"P(King | Face card) = {p_king_given_face:.4f}")

# 2

def bernoulli_pmf(k, p):
    return p if k == 1 else (1 - p)

def categorical_pmf(k, probs):
    return probs[k]

def poisson_pmf(k, lam):
    return (lam ** k) * math.exp(-lam) / factorial(k)

def uniform_pdf(x, a, b):
    if a <= x <= b:
        return 1.0 / (b - a)
    return 0.0

def normal_pdf(x, mu, sigma):
    coeff = 1.0 / (sigma * math.sqrt(2 * math.pi))
    exponent = -0.5 * ((x - mu) / sigma) ** 2
    return coeff * math.exp(exponent)

# 3

def expected_value(values, probabilities):
    return sum(v * p for v, p in zip(values, probabilities))

def variance(values, probabilities):
    mu = expected_value(values, probabilities)
    return sum([p * (mu - v) ** 2 for v, p in zip(values, probabilities)])

die_values = [1, 2, 3, 4, 5, 6]
die_probs = [1/6] * 6
mu = expected_value(die_values, die_probs)
var = variance(die_values, die_probs)
print(f"Die: E[X] = {mu:.4f}, Var(X) = {var:.4f}, SD = {var**0.5:.4f}")

# 4

def sample_bernoulli(p, n=1):
    return [1 if random.random() < p else 0 for _ in range(n)]

def sample_categorical(probs, n=1):
    cumulative = []
    total = 0
    for p in probs:
        total += p
        cumulative.append(total)
    samples = []
    for _ in range(n):
        r = random.random()
        for i, c in enumerate(cumulative):
            if r <= c:
                samples.append(i)
                break
    return samples

def sample_normal_box_muller(mu, sigma, n=1):
    samples = []
    for _ in range(n):
        u1 = random.random()
        u2 = random.random()
        z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
        samples.append(mu + sigma * z)
    return samples

# 5

def softmax(logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    exps = [math.exp(z) for z in shifted]
    total = sum(exps)
    return [e / total for e in exps]

def log_softmax(logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    log_sum_exp = max_logit + math.log(sum(math.exp(z) for z in shifted))
    return [z - log_sum_exp for z in logits]

def cross_enthropy_loss(logits, target_index):
    log_probs = log_softmax(logits)
    return - log_probs[target_index]

# 6

def demonstrate_clt(dist_fn, n_samples, n_averages):
    averages = []
    for _ in range(n_averages):
        samples = [dist_fn() for _ in range(n_samples)]
        averages.append(sum(samples) / len(samples))
    return averages

def sample_exponential(lam, n=1):
    samples = []
    for _ in range(n):
        u = random.random()
        x = -math.log(u) / lam
        samples.append(x)

    return samples

# samples = sample_exponential(1, 10000)
# print(sum(samples) / len(samples))

# import matplotlib.pyplot as plt
# import numpy as np

# plt.hist(samples, bins=50, density=True, label='sample')

# x = np.linspace(0, 8, 200)
# pdf = [1.0 * math.exp(-1.0 * xi) for xi in x]
# plt.plot(x, pdf, label='true PDF')

# plt.legend()
# plt.show()

die1 = [0.1, 0.1, 0.1, 0.1, 0.1, 0.5]  # heavily biased toward 6
die2 = [0.2, 0.2, 0.2, 0.2, 0.1, 0.1]  # biased toward low numbers

def joint_distribution(p1, p2):
    table = [[p1[i] * p2[j] for j in range(len(p2))] for i in range(len(p1))]
    return table

table = joint_distribution(die1, die2)

# for row in table:
#     print([round(v, 3) for v in row])

def marginal(table, axis):
    sum = []
    if axis == 1:
        for i in range(len(table[0])):
            local_sum = 0
            for row in table:
                local_sum += row[i]
            sum.append(local_sum)
    else:
        for i in range(len(table)):
            local_sum = 0
            for column in table[i]:
                local_sum += column
            sum.append(local_sum)

    return sum

m1 = marginal(table, 0)
m2 = marginal(table, 1)

print([round(v, 3) for v in m1])
print([round(v, 3) for v in m2])

print(die1)
print(die2)

def is_independent(table, m1, m2):
    for i in range(len(table)):
        for j in range(len(table[0])):
            if abs(table[i][j] - m1[i] * m2[j]) >= 1e-9:
                return False
    return True

print(is_independent(table, m1, m2))

for i in m1:
    print(i)

for j in m2:
    print(j)

logits = [2.0, 0.5, -1.0, 3.0, 0.1]
target = 3

loss = cross_enthropy_loss(logits, target)
print(f"Loss: {loss:.4f}")

import torch
import torch.nn as nn

logits_t = torch.tensor([[2.0, 0.5, -1.0, 3.0, 0.1]])
target_t = torch.tensor([3])

criterion = nn.CrossEntropyLoss()
loss_t = criterion(logits_t, target_t)
print(f"PyTorch loss: {loss_t.item():.4f}")

def sequence_probability(log_probs):
    return log_probs, sum(log_probs), math.exp(sum(log_probs))

log_probs = [math.log(0.01)] * 50
seq, total_log, raw_prob = sequence_probability(log_probs)

print(f"Total log probability: {total_log:.4f}")
print(f"Raw probability: {raw_prob:.2e}")