from functools import partial

from numpyro.handlers import substitute, trace
from numpyro.infer import util

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

    def get_trace(unconstrained_sample):
        """
        Transform a sample from unconstrained to constrained space, and then
        produce an updated model trace where the sites are updated to these values.

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
    
    if inv_temp > 0:
        # general case
        def tempered_pot(unconstrained_sample):
            log_prior = sum(
                site["fn"].log_prob(site["value"])
                for name,site in get_trace(unconstrained_sample).items()
                if site["type"] == "sample" and (
                    (not site["is_observed"]) or name.endswith("_log_det")
                )
            )
            log_lik = sum(
                site["fn"].log_prob(site["value"])
                for name,site in get_trace(unconstrained_sample).items()
                if site["type"] == "sample" and (
                    site["is_observed"] and (not name.endswith("_log_det"))
                )
            )
            return -(log_prior + inv_temp*log_lik)
        return tempered_pot
    
    else:
        # skip observed sites altogether
        def reference_pot(unconstrained_sample):
            return -sum(
                site["fn"].log_prob(site["value"])
                for name,site in get_trace(unconstrained_sample).items()
                if site["type"] == "sample" and (
                    (not site["is_observed"]) or name.endswith("_log_det")
                )
            )
        return reference_pot

def make_interpolator(model, model_args, model_kwargs):
    return partial(
        make_tempered_potential, 
        model, model_args=model_args,
        model_kwargs=model_kwargs
    )
