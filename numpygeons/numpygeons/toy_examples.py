from jax import lax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

##############################################################################
# classic Pigeons toy unidentifiable example
##############################################################################

def __toy_unid(n_flips, n_heads=None):
    p1 = numpyro.sample('p1', dist.Uniform())
    p2 = numpyro.sample('p2', dist.Uniform())
    p = numpyro.deterministic('p', p1*p2)
    numpyro.sample('n_heads', dist.Binomial(n_flips, p), obs=n_heads)

def toy_unid_example(n_heads=50, n_flips=100):
    model = __toy_unid
    model_args = (n_flips,)
    model_kwargs={"n_heads": n_heads}
    return (model, model_args, model_kwargs)

##############################################################################
# eight schools example
##############################################################################
 
def __eight_schools(sigma, y=None):
    mu = numpyro.sample('mu', dist.Normal(0, 5))
    tau = numpyro.sample('tau', dist.HalfCauchy(5))
    with numpyro.plate('school_plate', len(sigma)):
        theta = numpyro.sample('theta', dist.Normal(mu, tau))
        numpyro.sample('obs', dist.Normal(theta, sigma), obs=y)

def eight_schools_example():
    y = jnp.array([28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0])
    sigma = jnp.array([15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0])
    model = __eight_schools
    model_args = (sigma,)
    model_kwargs={'y': y}
    return (model, model_args, model_kwargs)

##############################################################################
# mRNA transfection example from Ballnus et al. (2017, dataset M1a)
# https://doi.org/10.1186/s12918-017-0433-1
##############################################################################

def _mrna(ts, ys=None):
    # priors
    lt0 = numpyro.sample('lt0', dist.Uniform(-2,1))
    lkm0 = numpyro.sample('lkm0', dist.Uniform(-5,5))
    lbeta = numpyro.sample('lbeta', dist.Uniform(-5,5))
    ldelta = numpyro.sample('ldelta', dist.Uniform(-5,5))
    lsigma = numpyro.sample('lsigma', dist.Uniform(-2,2))

    # transformed vars
    t0 = 10. ** lt0
    km0 = 10. ** lkm0
    beta = 10. ** lbeta
    delta = 10. ** ldelta
    sigma = 10. ** lsigma

    # likelihood: conditionally indep normals with mean
    #   m(t) = (km0/(delta-beta))[exp(-beta(t-t0)) - exp(-delta(t-t0))]
    # To avoid loss of precision from the `exp` difference, we can rewrite as
    #   (e^{-betaD}-e^{-deltaD}) = e^{-deltaD}expm1{(delta-beta)D})
    rel_ts = lax.max(jnp.zeros_like(ts), ts-t0)
    diff = delta-beta
    ms = km0*lax.exp(-delta*rel_ts)*(lax.expm1(diff*rel_ts)/diff)
    with numpyro.plate('dim', len(ts)):
        numpyro.sample('obs', dist.Normal(ms, scale=sigma), obs=ys)

def mrna():
    """
    mRNA transfection time series model (dataset M1a) described in 
    Ballnus et al. (2017).
    """
    import pandas as pd
    # Load the data
    dta = pd.read_csv(
        'https://raw.githubusercontent.com/Julia-Tempering/Pigeons.jl/refs/heads/main/examples/data/Ballnus_et_al_2017_M1a.csv',
        header=None
    )
    model = _mrna
    model_args = (jnp.array(dta[0]),)
    model_kwargs={'ys': jnp.array(dta[1])}
    return (model, model_args, model_kwargs)
