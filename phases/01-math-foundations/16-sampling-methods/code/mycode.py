import math
import random

random.seed(42)

def sample_uniform(a, b):
    return a + (b - a) * random.random()

def sample_exponential_inverse_cdf(lam):
    u = random.random()
    return -math.log(u) / lam

def verify_inverse_cdf(lam, n=10000):
    samples = [sample_exponential_inverse_cdf(lam) for _ in range(n)]
    empirical_mean = sum(samples) / len(samples)
    theoretical_mean = 1.0 / lam
    print(f" Exponential(lambda={lam}): empirical mean={empirical_mean:.4f},"
          f" theoretical mean={theoretical_mean:.4f}")
    return samples

def normal_pdf(x, mu, sigma):
    coeff = 1.0 / (sigma * math.sqrt(2 * math.pi))
    exponent = -0.5 * ((x - mu) / sigma) ** 2
    return coeff * math.exp(exponent)

def sample_normal_box_muller(mu=0.0, sigma=1.0):
    u1 = random.random()
    u2 = random.random()
    z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
    return mu + sigma * z

def rejection_sample(target_pdf, proposal_sample, proposal_pdf, M):
    attempts = 0
    while True:
        x = proposal_sample()
        u = random.random()
        attempts += 1
        if u < target_pdf(x) / (M * proposal_pdf(x)):
            return x, attempts

def rejection_sample_batch(target_pdf, proposal_sample, proposal_pdf, M, n):
    samples = []
    total_attempts = 0
    for _ in range(n):
        x, attempts = rejection_sample(target_pdf, proposal_sample, proposal_pdf, M)
        samples.append(x)
        total_attempts += attempts
    acceptance_rate = n / total_attempts
    return samples, acceptance_rate

