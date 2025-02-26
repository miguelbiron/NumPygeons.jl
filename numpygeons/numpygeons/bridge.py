from functools import partial

import jax
from jax import lax
from jax import numpy as jnp
jax.config.update('jax_platform_name', 'cpu') # DEBUG

from numpyro.handlers import seed, substitute, trace
from numpyro.infer import util

##############################################################################
# sampling utilities
##############################################################################

# sample from prior
def make_prior_sampler(model, model_args, model_kwargs, rng_key):
    traced_model = trace(seed(model, rng_seed=rng_key))
    unconstrain = partial(util.unconstrain_fn, model, model_args, model_kwargs)

    # note: can't jit this because it has side effects (rng_key is updated)
    def sample_iid():
        exec_trace = traced_model.get_trace(*model_args, **model_kwargs)
        params = {
            name: site["value"] for name, site in exec_trace.items() 
            if site["type"] == "sample" and not site["is_observed"]
        }
        return unconstrain(params)
    return sample_iid

# utility for creating a new sample state
def update_sample_field(kernel_state, sample_field, unconstrained_sample):
    return kernel_state._replace(**{sample_field: unconstrained_sample})

# sequentially sample multiple times, discard intermediate states
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

# def init_sample_container(constrained_sample, n_samples):
#     """
#     Initialize a pytree-compatible container for samples. The root of the tree
#     is a `dict` with the same structure as a constrained sample, with an
#     additional entry indicating the PT scan associated with a sample. This
#     allows us to merge containers produced by multiple processors in a 
#     distributed setting.

#     :param constrained_sample: A sample in constrained space.
#     :param n_samples: Number of samples that the container should hold.
#     :return: A `dict` for holding `n_samples` samples.
#     """
#     # create a prototypical entry for the container by copying the sample
#     # and adding a scan index to it
#     prototype_entry = constrained_sample.copy()
#     prototype_entry['__scan__'] = jnp.int32(0)

#     # build pytree from the prototype entry
#     def map_fn(array_value):
#         x = jnp.asarray(array_value)
#         return jnp.empty((n_samples, *x.shape), dtype=x.dtype)
#     return jax.tree.map(map_fn, prototype_entry)

# @jax.jit
# def add_sample_to_container(
#     container, 
#     constrained_sample, 
#     scan_idx, 
#     storage_idx
#     ):
#     """
#     Store a sample in the container.

#     :param container: A container generated by `init_sample_container`.
#     :param constrained_sample: A sample in constrained space.
#     :param scan_idx: Number of samples that the container should hold.
#     :return: A `dict` for holding `n_samples` samples.
#     """
#     constrained_sample['__scan__'] = jnp.int32(scan_idx)
#     def map_fn(container_array, new_val):
#         container_array = container_array.at[storage_idx].set(new_val)
#         return container_array
#     return jax.tree.map(map_fn, container, constrained_sample)

##############################################################################
# log density evaluation utilities
##############################################################################

# tempered potential constructor
# a.k.a. an interpolator for the tempered path of distributions
def make_tempered_potential(model, inv_temp, model_args, model_kwargs):
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

    :param model: A numpyro model.
    :param inv_temp: An inverse temperature (non-negative number).
    :param model_args: Model arguments.
    :param model_kwargs: Model keyword arguments.
    :return: A potential function for the tempered model.
    """

    @jax.jit
    def tempered_pot(unconstrained_sample):
        """
        Compute the tempered potential for an unconstrained sample. This involves
        two steps: we first constrain the sample and then use it to update the 
        model trace to match the sampled values. Then, we iterate the trace and
        compute the logprior -- including any logabsdetjac terms due to change of
        variables -- and loglikelihood. Finally, we return their tempered sum.

        :param unconstrained_sample: A sample from the model in unconstrained space.
        :return: The potential evaluated at the given sample.
        """
        sites = trace_from_unconst_samples(
            model, 
            unconstrained_sample, 
            model_args, 
            model_kwargs
        )
        log_prior = sum(
            site["fn"].log_prob(site["value"])
            for name,site in sites.items()
            if site["type"] == "sample" and (
                (not site["is_observed"]) or name.endswith("_log_det")
            )
        )
        log_lik = sum(
            site["fn"].log_prob(site["value"])
            for name,site in sites.items()
            if site["type"] == "sample" and (
                site["is_observed"] and (not name.endswith("_log_det"))
            )
        )
        return -(log_prior + inv_temp*log_lik)
    return tempered_pot

def make_interpolator(model, model_args, model_kwargs):
    return partial(
        make_tempered_potential, 
        model, model_args=model_args,
        model_kwargs=model_kwargs
    )
