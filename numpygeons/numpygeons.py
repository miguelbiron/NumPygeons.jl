from functools import partial

from numpyro.handlers import substitute, trace
from numpyro.infer import util

# tempering utilities
def make_tempered_potential(model, inv_temp, args, kwargs):
    def get_trace(unconstrained_sample):
        substituted_model = substitute(
            model, substitute_fn=partial(
                util._unconstrain_reparam, 
                unconstrained_sample
            )
        )
        return trace(substituted_model).get_trace(*args, **kwargs)
        
    if inv_temp > 0:
        # general case
        uno = inv_temp / inv_temp # dumb but maintains type
        def tempered_pot(unconstrained_sample):
            return -sum(
                site["fn"].log_prob(site["value"]) * (
                    inv_temp if (
                        site["is_observed"] and (not name.endswith("_log_det"))
                    ) else uno
                )
                for name,site in get_trace(unconstrained_sample).items()
                if site["type"] == "sample"
            )
        return tempered_pot
    else:
        # skip observed sites altogether
        def reference_pot(unconstrained_sample):
            return -sum(
                site["fn"].log_prob(site["value"])
                for name,site in get_trace(unconstrained_sample).items()
                if (
                    site["type"] == "sample" and
                    ((not site["is_observed"]) or name.endswith("_log_det"))
                )
            )
        return reference_pot
