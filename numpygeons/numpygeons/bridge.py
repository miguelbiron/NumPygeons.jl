from functools import partial

import jax
from jax import lax
from jax import numpy as jnp

from numpyro.handlers import seed, trace
from numpyro.infer import util

from autostep.statistics import AutoStepAdaptStats
from autostep.tempering import trace_from_unconst_samples

##############################################################################
# sampling utilities
##############################################################################

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

# make a new kernel state by refreshing the sample field from the prior
@partial(jax.jit, static_argnums=(0,3))
def sample_iid_kernel_state(
        model, 
        model_args, 
        model_kwargs, 
        sample_field, 
        kernel_state
    ):
    new_rng_key, iid_key = jax.random.split(kernel_state.rng_key)
    unconstrained_sample = sample_iid(model, model_args, model_kwargs, iid_key)
    return kernel_state._replace(
        rng_key=new_rng_key, **{sample_field: unconstrained_sample}
    )

# sequentially sample multiple times, discard intermediate states
# note: need partial jax.jit because 
#   - kernel is not a pytree
#   - `scan` needs `length` to be a constant. So n_refresh must be static
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

@partial(jax.jit, static_argnums=(0,))
def sample_extractor(
    model,
    model_args, 
    model_kwargs, 
    unconstrained_sample, 
    scan_idx
    ):
    """
    Takes an unconstrained sample, transforms it into constrained space, 
    executes a model trace to possibly recover additional deterministic values,
    and finally returns a dictionary with values for all latent sites.

    :param model: A numpyro model.
    :param model_args: Model arguments.
    :param model_kwargs: Model keyword arguments.
    :param unconstrained_sample: A sample in unconstrained space.
    :param scan_idx: The index of the scan that generated the sample.
    :return: A `dict`.
    """
    exec_trace = trace_from_unconst_samples(
        model, 
        model_args, 
        model_kwargs,
        unconstrained_sample
    )
    sample_values = {
        name: site["value"] 
        for name, site in exec_trace.items() 
        if (site["type"] == "sample" and not site["is_observed"]) or
        (site["type"] == "deterministic")
    }
    sample_values['__scan__'] = scan_idx
    return sample_values


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
