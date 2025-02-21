from functools import partial

import jax
jax.config.update('jax_platform_name', 'cpu')

from jax import lax

from numpyro.handlers import substitute, trace
from numpyro.infer import util

#######################################
# sampling utilities
#######################################

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

#######################################
# log density evaluation utilities
#######################################

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
        :return: A trace.
        """
        substituted_model = substitute(
            model, substitute_fn=partial(
                util._unconstrain_reparam, 
                unconstrained_sample
            )
        )
        sites = trace(substituted_model).get_trace(*model_args, **model_kwargs)
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
