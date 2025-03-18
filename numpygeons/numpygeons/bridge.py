from functools import partial

import jax
from jax import lax
from jax import numpy as jnp

from numpyro.handlers import seed, substitute, trace
from numpyro.infer import util
from numpyro.distributions.util import is_identically_one

from autostep.statistics import AutoStepAdaptStats

##############################################################################
# sampling utilities
##############################################################################

# need these only because destructuring dictionaries within julia is impossible
def make_kernel_from_model(model, kernel_type, kernel_kwargs):
    return kernel_type(model, **kernel_kwargs)
def make_kernel_from_potential(potential_fn, kernel_type, kernel_kwargs):
    return kernel_type(potential_fn=potential_fn, **kernel_kwargs)
def update_sample_field(kernel_state, sample_field, unconstrained_sample):
    return kernel_state._replace(**{sample_field: unconstrained_sample})

# sample from prior
@partial(jax.jit, static_argnums=(0,))
def sample_iid(model, model_args, model_kwargs, rng_key):
    traced_model = trace(seed(model, rng_seed=rng_key))
    unconstrain = partial(util.unconstrain_fn, model, model_args, model_kwargs)
    exec_trace = traced_model.get_trace(*model_args, **model_kwargs)
    params = {
        name: site["value"] for name, site in exec_trace.items() 
        if site["type"] == "sample" and not site["is_observed"]
    }
    return unconstrain(params)

def make_prior_sampler(model, model_args, model_kwargs):
    return partial(sample_iid, model, model_args, model_kwargs)

# sequentially sample multiple times, discard intermediate states
# note: need partial jax.jit because 
#   - kernel is not a pytree
#   - `scan` needs `length` to be a constant. So n_refresh must be static
# Since n_refresh doesn't change, this means that this is recompiled 
# (n_chains-1) times per round, because all local kernels change at the 
# beginning of the round (due to new tempered potential function)
@partial(jax.jit, static_argnums=(0,2))
def loop_sample(kernel, init_state, n_refresh, model_args, model_kwargs):
    """
    Performs `n_refresh` Markov steps with the provided kernel, returning
    only the last state of the chain.
    
    :param kernel: An instance of `numpyro.infer.MCMC`.
    :param init_state: Starting point of the sampler.
    :param n_refresh: Number of Markov steps to take with the sampler.
    :param model_args: Model arguments.
    :param model_kwargs: Model keyword arguments.
    :return: The last state of the chain.
    """
    return lax.scan(
        lambda state, _: (kernel.sample(state,model_args,model_kwargs), None), 
        init_state,
        length=n_refresh
    )[0]
 
##############################################################################
# recorder utilities
##############################################################################

def trace_from_unconst_samples(
        model, 
        unconstrained_sample, 
        model_args, 
        model_kwargs
    ):
    """
    Constrain the sample and then use it to update the model trace to match the
    sampled values.

    :param unconstrained_sample: A sample from the model in unconstrained space.
    :return: A trace.
    """
    substituted_model = substitute(
        model, substitute_fn=partial(
            util._unconstrain_reparam, 
            unconstrained_sample
        )
    )
    return trace(substituted_model).get_trace(*model_args, **model_kwargs)


def make_sample_extractor(
        model, 
        model_args, 
        model_kwargs, 
        capture_deterministic=True
    ):
    """
    Build a function that takes an unconstrained sample, transforms it into
    constrained space, executes a model trace to possibly recover additional
    deterministic values, and finally returns a dictionary with values for
    all latent sites.

    :param model: A numpyro model.
    :param model_args: Model arguments.
    :param model_kwargs: Model keyword arguments.
    :param capture_deterministic: Should deterministic fields be retrieved?
    :return: A function.
    """
        
    @jax.jit
    def sample_extractor(unconstrained_sample, scan_idx):
        exec_trace = trace_from_unconst_samples(
            model, 
            unconstrained_sample, 
            model_args, 
            model_kwargs
        )
        sample_values = {
            name: site["value"] 
            for name, site in exec_trace.items() 
            if (site["type"] == "sample" and not site["is_observed"]) or
            (capture_deterministic and (site["type"] == "deterministic"))
        }
        sample_values['__scan__'] = jnp.int32(scan_idx)
        return sample_values

    return sample_extractor

@jax.jit
def stack_samples(samples_list):
    samples_dict = jax.tree.map(lambda *v: jnp.stack(v), *samples_list)
    sorting_idxs = samples_dict['__scan__'].argsort()
    return {name: vals[sorting_idxs] for name, vals in samples_dict.items()}

@jax.jit
def merge_adapt_stats(as1, as2):
    if not isinstance(as1, AutoStepAdaptStats):
        raise(
            NotImplementedError(
                f"""
                Don't know how to merge {type(as1)} objects.
                """
            )
        )
    assert type(as1) == type(as2), f"""
    Incompatible types: `as1` is a {type(as1)}, but `as2` is a {type(as2)}.
    """
    sample_idx = as1.sample_idx + as2.sample_idx
    frac1 = as1.sample_idx / sample_idx
    frac2 = 1-frac1
    mean_step_size = frac1*as1.mean_step_size + frac2*as2.mean_step_size
    mean_acc_prob = frac1*as1.mean_acc_prob + frac2*as2.mean_acc_prob
    means_flat = frac1*as1.means_flat + frac2*as2.means_flat
    vars_flat = frac1*as1.vars_flat + frac2*as2.vars_flat
    return AutoStepAdaptStats(
        sample_idx, mean_step_size, mean_acc_prob, means_flat, vars_flat
    )

@jax.jit
def swap_adapt_stats(old_kernel_state, new_adapt_stats):
    assert isinstance(new_adapt_stats, AutoStepAdaptStats)
    old_adapt_stats = old_kernel_state.stats.adapt_stats
    new_kernel_state = old_kernel_state._replace(
        stats = old_kernel_state.stats._replace(adapt_stats = new_adapt_stats)
    )
    return (new_kernel_state, old_adapt_stats)

##############################################################################
# log density evaluation utilities
# slight modification of numpyro.infer.util.compute_log_probs
##############################################################################

def log_prob_site(site):
    value = site["value"]
    intermediates = site["intermediates"]
    scale = site["scale"]
    if intermediates:
        log_prob = site["fn"].log_prob(value, intermediates)
    else:
        guide_shape = jnp.shape(value)
        model_shape = tuple(
            site["fn"].shape()
        )  # TensorShape from tfp needs casting to tuple
        try:
            lax.broadcast_shapes(guide_shape, model_shape)
        except ValueError:
            raise ValueError(
                "Model and guide shapes disagree at site: '{}': {} vs {}".format(
                    site["name"], model_shape, guide_shape
                )
            )
        log_prob = site["fn"].log_prob(value)

    log_prob_sum = jnp.sum(log_prob)

    if (scale is not None) and (not is_identically_one(scale)):
        log_prob_sum = scale * log_prob_sum
    
    return log_prob_sum

def is_observation(site):
    return site["is_observed"] and (not site["name"].endswith("_log_det"))

def log_prior(model_trace):
    return sum(
        log_prob_site(site) 
        for site in model_trace.values()
        if site["type"] == "sample" and (not is_observation(site))
    )

def log_lik(model_trace):
    return sum(
        log_prob_site(site) 
        for site in model_trace.values()
        if site["type"] == "sample" and is_observation(site)
    )

# tempered potential constructor
# a.k.a. an interpolator for the tempered path of distributions
# note: as in the case of the loop sampler, the `tempered_pot` is recompiled
# (n_chains-1) times per round. This is much faster however.
@partial(jax.jit, static_argnums=(0,))
def tempered_pot(model, model_args, model_kwargs, inv_temp, unconstrained_sample):
    """
    Build a tempered version of the potential function associated with the
    posterior distribution of a numpyro model. Specifically,

    .. code-block:: python
        tempered_potential(x) = -log(pi_beta(x))

    where `beta` is the inverse temperature, and

    .. code-block:: python
        pi_beta(x) = prior(x) * likelihood(x) ** beta

    Hence, when `inv_temp=0`, the tempered model reduces to the prior, whereas
    for `inv_temp=1`, the original posterior distribution is recovered.

    To achieve this, we first constrain the sample and then use it to update the 
    model trace to match the sampled values. Then, we iterate the trace and
    compute the logprior -- including any logabsdetjac terms due to change of
    variables -- and loglikelihood. Finally, we return their tempered sum.

    :param model: A numpyro model.
    :param model_args: Model arguments.
    :param model_kwargs: Model keyword arguments.
    :param inv_temp: An inverse temperature (non-negative number).
    :param unconstrained_sample: A sample from the model in unconstrained space.
    :return: The potential evaluated at the given sample.
    """
    model_trace = trace_from_unconst_samples(
        model, 
        unconstrained_sample, 
        model_args, 
        model_kwargs
    )
    return -(log_prior(model_trace) + inv_temp*log_lik(model_trace))
   

def make_tempered_potential(model, model_args, model_kwargs, inv_temp):
    return partial(tempered_pot, model, model_args, model_kwargs, inv_temp)

def make_interpolator(model, model_args, model_kwargs):
    return partial(make_tempered_potential, model, model_args, model_kwargs)
