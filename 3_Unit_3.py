import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from utils import app_header, quiz_block, ensure_seed, progress_sidebar, figure_show

st.set_page_config(page_title="Unit 3 — Distributions + Simulation", layout="wide")
progress_sidebar(active_unit=3)
with st.sidebar:
    seed = st.number_input("Seed", value=42, step=1)
ensure_seed(seed)

app_header("UNIT 3: Probability Distributions + Simulation",
           "PMFs, PDFs/CDFs, expectation/variance, MLE, Monte Carlo π, CLT.")

st.subheader("3.1 Discrete PMFs — Binomial & Poisson")
p = st.slider("Binomial p", 0.05, 0.95, 0.3, 0.01)
ntr = st.slider("Binomial n", 1, 40, 10, 1)
k = np.arange(0, ntr+1); binom_pmf = stats.binom.pmf(k, n=ntr, p=p)
fig, ax = plt.subplots(); ax.stem(k, binom_pmf); ax.set_title("Binomial PMF"); figure_show(fig)

lam = st.slider("Poisson λ", 0.1, 12.0, 3.0, 0.1)
k2 = np.arange(0, int(max(12, lam+6))+1); pois_pmf = stats.poisson.pmf(k2, mu=lam)
fig, ax = plt.subplots(); ax.stem(k2, pois_pmf); ax.set_title("Poisson PMF"); figure_show(fig)

st.markdown("---")
st.subheader("3.2 Continuous PDFs & CDFs — Normal / Uniform / Exponential")
x = np.linspace(-4, 4, 400)
mu = st.slider("Normal μ", -2.0, 2.0, 0.0, 0.1)
sigma = st.slider("Normal σ", 0.2, 3.0, 1.0, 0.1)
pdf = stats.norm.pdf(x, mu, sigma); cdf = stats.norm.cdf(x, mu, sigma)
fig, ax = plt.subplots(); ax.plot(x, pdf); ax.set_title("Normal PDF"); figure_show(fig)
fig, ax = plt.subplots(); ax.plot(x, cdf); ax.set_title("Normal CDF"); figure_show(fig)
u = np.linspace(0,1,200); fig, ax = plt.subplots(); ax.plot(u, stats.uniform.pdf(u,0,1)); ax.set_title("Uniform(0,1) PDF"); figure_show(fig)
t = np.linspace(0,5,300); fig, ax = plt.subplots(); ax.plot(t, stats.expon.pdf(t, scale=1)); ax.set_title("Exponential(λ=1) PDF"); figure_show(fig)

st.subheader("3.3 Expectation & Variance — Simulation vs Theory")
N = st.slider("Sample size", 1000, 200000, 50000, 1000)
samp = np.random.normal(mu, sigma, size=N)
fig, ax = plt.subplots(); ax.hist(samp, bins=50, density=True, alpha=0.7)
xs = np.linspace(mu-4*sigma, mu+4*sigma, 300); ax.plot(xs, stats.norm.pdf(xs, mu, sigma))
ax.set_title("Simulation vs Theoretical Normal"); figure_show(fig)
st.write("Empirical mean, var:", float(samp.mean()), float(samp.var()))

st.markdown("---")
st.subheader("3.4 MLE: Normal & Binomial")
data = np.random.normal(loc=1.0, scale=2.0, size=1000)
mu_hat = data.mean(); sigma_hat = data.std(ddof=0)
st.write("Normal MLE → μ̂ =", float(mu_hat), " σ̂ =", float(sigma_hat))
n_trials = 10
obs_successes = np.random.binomial(n_trials, 0.45, size=200)
p_hat = obs_successes.sum() / (n_trials * len(obs_successes))
st.write("Binomial MLE → p̂ =", float(p_hat))

st.markdown("---")
st.subheader("3.5 Monte Carlo: π & CLT")
N = st.slider("π — number of points", 1000, 300000, 100000, 1000)
pts = np.random.rand(N, 2); inside = (pts[:,0]**2 + pts[:,1]**2) <= 1.0
pi_est = 4*inside.mean(); st.write("Estimate of π ≈", float(pi_est))
draws = st.slider("CLT — draws", 1000, 30000, 10000, 1000)
sample_size = st.slider("CLT — sample size", 2, 200, 20, 1)
means = np.mean(np.random.exponential(scale=1.0, size=(draws, sample_size)), axis=1)
fig, ax = plt.subplots(); ax.hist(means, bins=50, density=True, alpha=0.7)
xs = np.linspace(0,2,300); ax.plot(xs, stats.norm.pdf(xs, loc=1.0, scale=1.0/np.sqrt(sample_size)))
ax.set_title("CLT: Means of Exponential Samples"); figure_show(fig)

quiz_block(
    ["PMF vs PDF?", "What does CLT say?", "Effect of λ on Poisson?"],
    ["PMF is discrete; PDF is continuous (integrates to 1).",
     "Means of i.i.d. samples are ~Normal for large n.",
     "Mean & variance increase with λ; mass shifts right."]
)