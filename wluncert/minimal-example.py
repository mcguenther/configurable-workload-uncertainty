import itertools
import arviz as az
import jax.numpy as jnp
import numpyro
import numpyro.distributions as npdist
import scipy.stats
import random as python_random
from jax import random
from matplotlib import pyplot as plt
from numpyro.handlers import condition as npcondition
from numpyro.infer import MCMC as npMCMC, NUTS as npNUTS, BarkerMH as npBMH, HMC as npHMC, SA
from sklearn.preprocessing import StandardScaler, MinMaxScaler

numpyro.set_host_device_count(4)


def model(a, b, c, obs=None):
    mean = 0
    stddev = 1
    influence_a = numpyro.sample("influence_a", npdist.Normal(mean, stddev))
    influence_b = numpyro.sample("influence_b", npdist.Normal(mean, stddev))
    influence_c = numpyro.sample("influence_c", npdist.Normal(mean, stddev))
    # base = numpyro.sample("base", npdist.HalfNormal(2.0))
    base = numpyro.sample("base", npdist.Normal(0, 2.0))
    result = base + a * influence_a + b * influence_b + c * influence_c
    error_stddev = numpyro.sample("error", npdist.Exponential(0.1))
    # sigma = numpyro.sample('sigma', npdist.Exponential(1.))
    with numpyro.plate("data", len(a)):
        obs = numpyro.sample("nfp", npdist.Normal(result, error_stddev), obs=obs)
    return obs


def ask_oracle_2(config):
    base = 20
    a, b, c = config
    influence_a = 5
    influence_a = influence_a * scipy.stats.norm(1.0, 1.0).rvs(1)[0]
    if c:
        influence_b = 10
    else:
        influence_b = 0.5
    influence_c = 2
    exact_nfp = influence_a * a + influence_b * b + influence_c * c + base
    noise = scipy.stats.norm(1, 0.001).rvs(1)[0]
    nfp = exact_nfp * noise
    return nfp


def ask_oracle(config):
    base = 20
    a, b, c = config
    influence_a = float(scipy.stats.norm(5, 3).rvs(1)[0])
    influence_b = 0.5
    influence_c = 8
    nfp = influence_a * a + influence_b * b + influence_c * c + base
    return nfp


def main():
    configs = list(itertools.product([True, False], repeat=3))
    configs = configs * 30  # simulating repeated measurements
    python_random.shuffle(configs)
    nfp = jnp.atleast_1d(list(map(ask_oracle, configs)))
    X = jnp.atleast_2d(configs)
    X = jnp.array(MinMaxScaler().fit_transform(X))
    nfp = jnp.array(MinMaxScaler().fit_transform(jnp.atleast_2d(nfp).T)[:, 0])
    # nuts_kernel = npNUTS(model, target_accept_prob=0.9,max_tree_depth=20)
    nuts_kernel = npNUTS(model, target_accept_prob=0.9,
                         max_tree_depth=20, regularize_mass_matrix=False)
    # nuts_kernel = npHMC(model, )
    n_chains = 3
    mcmc = npMCMC(nuts_kernel, num_samples=4000,
                  num_warmup=7000, progress_bar=False,
                  num_chains=n_chains, )
    rng_key = random.PRNGKey(10)
    mcmc.run(rng_key, X[:, 0], X[:, 1], X[:, 2], obs=nfp)

    mcmc.print_summary()
    az_data = az.from_numpyro(mcmc, num_chains=n_chains)
    print()
    print("ESS")
    print(az.ess(az_data))
    print()
    print("MCSE")
    print(az.mcse(az_data))

    az.plot_autocorr(az_data, combined=True)
    plt.suptitle("Auto Correlation")
    plt.tight_layout()
    plt.show()

    az.plot_ess(az_data, kind="local")
    plt.suptitle("Effective Sample Size Local")
    plt.tight_layout()
    plt.show()

    az.plot_ess(az_data, kind="quantile")
    plt.suptitle("Effective Sample Size Quantile")
    plt.tight_layout()
    plt.show()

    az.plot_trace(az_data, legend=True, )

    plt.tight_layout()
    plt.savefig("toy-abc-example-trace.pdf")
    plt.savefig("toy-abc-example-trace.png")
    plt.show()
    return


if __name__ == '__main__':
    main()
