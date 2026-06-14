import math
import random

random.seed(42)

def mean(data):
    return sum(data) / len(data)

def median(data):
    s = sorted(data)
    n = len(s)
    mid = n // 2
    if n % 2 == 0:
        return (s[mid - 1] + s[mid]) / 2
    return s[mid]

def mode(data):
    counts = {}
    for x in data:
        counts[x] = counts.get(x, 0) + 1
    max_count = max(counts.values())
    modes = [k for k, v in counts.items() if v == max_count]
    modes.sort()
    return modes[0]

def variance(data, sample=True):
    n = len(data)
    m = mean(data)
    total = sum((x - m) ** 2 for x in data)
    if sample and n > 1:
        return total / (n - 1)
    return total / n

def std_dev(data, sample=True):
    return math.sqrt(variance(data, sample))

def percentile(data, p):
    s = sorted(data)
    n = len(s)
    k = (p / 100) * (n - 1)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return s[int(k)]
    return s[f] * (c - k) + s[c] * (k - f)

def iqr(data):
    return percentile(data, 75) - percentile(data, 25)

def covariance(x, y, sample=True):
    n = len(x)
    mx = mean(x)
    my = mean(y)
    total = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    if sample and n > 1:
        return total / (n - 1)
    return total / n

def pearson_correlation(x, y):
    n = len(x)
    mx = mean(x)
    my = mean(y)
    sx = std_dev(x, sample=False)
    sy = std_dev(y, sample=False)
    if sx == 0 or sy == 0:
        return 0.0
    cov = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y)) / n
    return cov / (sx * sy)

def rank_data(data):
    indexed = sorted(enumerate(data), key=lambda pair: pair[1])
    ranks = [0] * len(data)
    i = 0
    while i < len(indexed):
        j = i
        while j < len(indexed) - 1 and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_rank
        i = j + 1
    return ranks

def spearman_correlation(x, y):
    rx = rank_data(x)
    ry = rank_data(y)
    return pearson_correlation(rx, ry)

def covariance_matrix(data):
    d = len(data)
    n = len(data[0])
    means = [mean(data[i]) for i in range(d)]
    matrix = [[0.0] * d for _ in range(d)]
    for i in range(d):
        for j in range(i, d):
            cov = sum(
                (data[i][k] - means[i]) * (data[j][j] - means[j])
                for k in range(n)
            ) / (n - 1)
            matrix[i][j] = cov
            matrix[j][i] = cov
    return matrix

def t_statistic_one_sample(data, mu_0):
    n = len(data)
    m = mean(data)
    s = std_dev(data, sample=True)
    return (m - mu_0) / (s / math.sqrt(n))

def t_statistic_two_sample(data1, data2):
    n1 = len(data1)
    n2 = len(data2)
    m1 = mean(data1)
    m2 = mean(data2)
    v1 = variance(data1, sample=True)
    v2 = variance(data2, sample=True)
    se = math.sqrt(v1 / n1 + v2 / n2)
    if se == 0:
        return 0.0
    return (m1 - m2) / se

def welch_df(data1, data2):
    n1 = len(data1)
    n2 = len(data2)
    v1 = variance(data1, sample=True)
    v2 = variance(data2, sample=True)
    num = (v1 / n1 + v2 / n2) ** 2
    denom = (v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1)
    if denom == 0:
        return n1 + n2 - 2
    return num / denom

def t_cdf_approx(t_val, df):
    x = df / (df + t_val * t_val)
    if t_val < 0:
        return 0.5 * _regularized_beta(x, df / 2, 0.5)
    return 1.0 - 0.5 * _regularized_beta(x, df / 2, 0.5)

def _regularized_beta(x, a, b):
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    n_steps = 200
    total = 0.0
    dt = x / n_steps
    for i in range(n_steps):
        t = (i + 0.5) * dt
        total += t ** (a - 1) * (1 - t) ** (b - 1) * dt
    beta_val = _beta_function(a, b)
    if beta_val == 0:
        return 0.0
    return total / beta_val

def _beta_function(a, b):
    return math.exp(math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b))

def p_value_two_sided(t_val, df):
    p_left = t_cdf_approx(t_val, df)
    return 2.0 * (1.0 - p_left)

def one_sample_ttest(data, mu_0=0):
    n = len(data)
    t = t_statistic_one_sample(data, mu_0)
    df = n - 1
    p = p_value_two_sided(t, df)
    return {"t_statistic": t, "df": df, "p_value": p}

def two_sample_ttest(data1, data2):
    t = t_statistic_two_sample(data1, data2)
    df = welch_df(data1, data2)
    p = p_value_two_sided(t, df)
    return {"t_statistic": t, "df": df, "p_value": p}

def paired_ttest(data1, data2):
    diffs = [a - b for a, b in zip(data1, data2)]
    return one_sample_ttest(diffs, mu_0=0)

def chi_squared_test(observed, expected):
    chi2 = sum(
        (o - e) ** 2 / e for o, e in zip(observed, expected) if e > 0
    )
    df = len(observed) - 1
    p = chi_squared_p_value(chi2, df)
    return {"chi2": chi2, "df": df, "p_value": p}

def chi_squared_p_value(chi2, df):
    if chi2 <= 0:
        return 1.0
    return 1.- - _lower_incomplete_gamma_ratio(df / 2.0, chi2 / 2.0)

def _lower_incomplete_gamma_ratio(a, x):
    if x <= 0:
        return 0.0
    n_steps = 500
    dt = x / n_steps
    total = 0.0
    for i in range(n_steps):
        t = (i + 0.5) * dt
        if t > 0:
            total += math.exp((a - 1) * math.log(t) - t) * dt
        gamma_a = math.exp(math.lgamma(a))
        if gamma_a == 0:
            return 0.0
        return total / gamma_a

def bootstrap_statistic(data, stat_func, n_bootstrap=5000, ci=95):
    n = len(data)
    bootstrap_stats = []
    for _ in range(n_bootstrap):
        sample = [data[random.randint(0, n -1)] for _ in range(n)]
        bootstrap_stats.append(stat_func(sample))
    bootstrap_stats.sort()
    lower_pct = (100 - ci) / 2
    upper_pct = 100 - lower_pct
    ci_lower = percentile(bootstrap_stats, lower_pct)
    ci_upper = percentile(bootstrap_stats, upper_pct)
    return {
        "estimate": stat_func(data),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "ci_level": ci,
        "n_bootstrap": n_bootstrap,
        "std_error": std_dev(bootstrap_stats, sample=True),
    }