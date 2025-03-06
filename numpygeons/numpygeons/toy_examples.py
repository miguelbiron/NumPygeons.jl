import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

##############################################################################
# classic Pigeons toy unidentifiable example
##############################################################################

def toy_unid(n_flips, n_heads=None):
    p1 = numpyro.sample('p1', dist.Uniform())
    p2 = numpyro.sample('p2', dist.Uniform())
    p = numpyro.deterministic('p', p1*p2)
    numpyro.sample('n_heads', dist.Binomial(n_flips, p), obs=n_heads)

def toy_unid_example(n_heads=50, n_flips=100):
    model = toy_unid
    model_args = (n_flips,)
    model_kwargs={"n_heads": n_heads}
    return (model, model_args, model_kwargs)

##############################################################################
# eight schools example
##############################################################################
 
def eight_schools(sigma, y=None):
    mu = numpyro.sample('mu', dist.Normal(0, 5))
    tau = numpyro.sample('tau', dist.HalfCauchy(5))
    with numpyro.plate('school_plate', len(sigma)):
        theta = numpyro.sample('theta', dist.Normal(mu, tau))
        numpyro.sample('obs', dist.Normal(theta, sigma), obs=y)

def eight_schools_example():
    y = jnp.array([28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0])
    sigma = jnp.array([15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0])
    model = eight_schools
    model_args = (sigma,)
    model_kwargs={'y': y}
    return (model, model_args, model_kwargs)
