import numpyro
import numpyro.distributions as dist

# define a model
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
