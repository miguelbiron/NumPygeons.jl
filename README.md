# NumPygeons.jl

[![Build Status](https://github.com/miguelbiron/NumPygeons.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/miguelbiron/NumPygeons.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/miguelbiron/NumPygeons.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/miguelbiron/NumPygeons.jl)

Sample [NumPyro](https://num.pyro.ai/) models using Parallel Tempering with the
julia package [Pigeons](https://pigeons.run/dev/) via 
[juliacall](https://juliapy.github.io/PythonCall.jl/dev/juliacall/).

**Important**: The project is still in early development, and therefore the 
package has not been registered yet.

## Usage from Python

Although it is possible to use NumPygeons from julia, this package is intended 
to be used from Python.


### Prerequisites

Start by installing julliacall. Assuming you have a pip-managed virtual 
environment, this is as easy as

```python
pip install juliacall
```

**Optional**: if you want to run your NumPyro models on an accelerator (GPU/TPU),
make sure to 
[install the correct version of JAX](https://jax.readthedocs.io/en/latest/installation.html).
Otherwise, the following will install the default, CPU-only version of JAX


### Installation

Start a Python REPL and run

```python
from juliacall import Main as jl
jl.seval("""
using Pkg
Pkg.add([
    (;name="Pigeons", rev="main"),
    (;url="https://github.com/miguelbiron/NumPygeons.jl.git")
])
""")
```

If this is your first time using juliacall, it will create a fresh julia
project inside your python virtual environment folder. Then we use regular
julia code to add both Pigeons (latest development version) and NumPygeons.


### Example

Start by loading juliacall and the required Julia packages 
```python
from juliacall import Main as jl
jl.seval("using Pigeons, NumPygeons")
```
The first time you load NumPygeons, it will install an internal bridge python
package to your virtual environment.

Let's now run the 8 schools example featured on the [NumPyro 
"getting started" page](https://num.pyro.ai/en/stable/getting_started.html).

```python
import jax.numpy as jnp

# define the model
def eight_schools(sigma, y=None):
    # "wait, module imports inside the model"?? Yes! They are necessary for
    # distributed sampling. Scroll down to find out more. 
    import numpyro
    import numpyro.distributions as dist
    mu = numpyro.sample('mu', dist.Normal(0, 5))
    tau = numpyro.sample('tau', dist.HalfCauchy(5))
    with numpyro.plate('school_plate', len(sigma)):
        theta = numpyro.sample('theta', dist.Normal(mu, tau))
        numpyro.sample('obs', dist.Normal(theta, sigma), obs=y)

# define the data
y = jnp.array([28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0])
sigma = jnp.array([15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0])
```

Having defined the model, running Pigeons is easy. By default, NumPygeons uses
the [AutoMALA](https://proceedings.mlr.press/v238/biron-lattes24a.html) MCMC kernel
from the [autostep package](https://github.com/UBC-Stat-ML/autostep). 
```python
# run Pigeons
path = jl.NumPyroPath(
    model = eight_schools,
    model_args = (sigma,),
    model_kwargs={'y': y},
)
pt = jl.pigeons(target=path, n_chains=3)

┌ Info: Neither traces, disk, nor online recorders included.
│    You may not have access to your samples (unless you are using a custom recorder, or maybe you just want log(Z)).
└    To add recorders, use e.g. pigeons(target = ..., record = [traces; record_default()])
────────────────────────────────────────────────────────────────────────────
  scans        Λ        time(s)    allc(B)  log(Z₁/Z₀)   min(α)     mean(α) 
────────── ────────── ────────── ────────── ────────── ────────── ──────────
        2      0.913       1.88   8.28e+03      -35.4     0.0871      0.544 
        4     0.0313       1.61   1.57e+04      -32.2      0.969      0.984 
        8      0.676       1.78   2.83e+04      -31.3      0.329      0.662 
       16      0.869       1.53   5.11e+04      -31.2      0.273      0.565 
       32       1.05       1.53   9.91e+04      -31.9      0.281      0.473 
       64      0.991       1.49   1.94e+05      -32.5      0.399      0.504 
      128       1.02       1.67   3.82e+05      -31.2      0.433      0.489 
      256      0.853       1.94    7.6e+05      -30.8      0.529      0.573 
      512      0.974       2.04   1.52e+06      -31.1      0.468      0.513 
 1.02e+03      0.855       2.86   3.03e+06      -31.2      0.534      0.572 
────────────────────────────────────────────────────────────────────────────
```

#### Traces 

NumPygeons has a specialized recorder to capture samples from 
the target without converting them from JAX arrays. To use it, just add it to
the list of recorders like so (we add other interesting recorders too)
```python
recorders = jl.seval("[record_default();energy_ac1;round_trip;numpyro_trace]")
pt = jl.pigeons(target=path, n_chains=3, record=recorders)

┌ Info: Neither traces, disk, nor online recorders included.
│    You may not have access to your samples (unless you are using a custom recorder, or maybe you just want log(Z)).
└    To add recorders, use e.g. pigeons(target = ..., record = [traces; record_default()])
─────────────────────────────────────────────────────────────────────────────────────────────────────────────
  scans     restarts      Λ        time(s)    allc(B)  log(Z₁/Z₀)   min(α)     mean(α)    max|ρ|     mean|ρ| 
────────── ────────── ────────── ────────── ────────── ────────── ────────── ────────── ────────── ──────────
        2          0      0.913       1.56   1.26e+06      -35.4     0.0871      0.544          1          1 
        4          1     0.0313       1.46   2.72e+04      -32.2      0.969      0.984      0.991      0.991 
        8          1      0.676       1.47   4.52e+04      -31.3      0.329      0.662      0.993      0.983 
       16          2      0.869       1.49   7.87e+04      -31.2      0.273      0.565      0.992      0.951 
       32          5       1.05       1.48   1.49e+05      -31.9      0.281      0.473      0.995      0.995 
       64         10      0.991        1.6   2.86e+05      -32.5      0.399      0.504      0.987      0.986 
      128         12       1.02        1.7   5.61e+05      -31.2      0.433      0.489      0.989      0.981 
      256         44      0.853       1.71   1.11e+06      -30.8      0.529      0.573      0.991       0.99 
      512         59      0.974        2.1   2.21e+06      -31.1      0.468      0.513      0.991      0.988 
 1.02e+03        167      0.855       2.86   4.41e+06      -31.2      0.534      0.572      0.992      0.992 
─────────────────────────────────────────────────────────────────────────────────────────────────────────────
```

We can then for example inspect the data using arviz
```python
import arviz as az

# add a singleton "chain" dimension to conform to arviz specification
az_samples=az.from_dict(
  jax.tree.map(lambda x: jax.numpy.expand_dims(x, axis=0), samples)
)
az.summary(az_samples,var_names=["mu","tau","theta"])

arviz - WARNING - Shape validation failed: input_shape: (1, 1024), minimum_shape: (chains=2, draws=4)
           mean     sd  hdi_3%  hdi_97%  mcse_mean  mcse_sd  ess_bulk  ess_tail  r_hat
mu        3.761  2.925  -1.398    9.401      0.278    0.155     112.0     134.0    NaN
tau       3.591  3.119   0.005    9.653      0.352    0.202      75.0     131.0    NaN
theta[0]  5.355  4.979  -3.462   15.836      0.668    0.571      72.0      88.0    NaN
theta[1]  4.089  4.015  -2.743   11.262      0.352    0.386     122.0     186.0    NaN
theta[2]  2.884  5.683 -11.355   11.409      1.682    0.682      12.0      40.0    NaN
theta[3]  4.345  4.171  -2.715   11.773      0.312    0.311     154.0     186.0    NaN
theta[4]  3.996  4.585  -3.287   13.935      0.379    0.321     146.0     103.0    NaN
theta[5]  4.095  5.142  -4.138   15.223      0.581    0.429      80.0     128.0    NaN
theta[6]  5.593  5.098  -2.500   15.687      0.572    0.699      82.0      90.0    NaN
theta[7]  4.037  5.412  -6.516   13.515      0.574    0.322      79.0     102.0    NaN
```

#### Distributed sampling

The way we wrote the model makes it so that distributed sampling can be easily
achieved by running almost the same code you would use from Julia. The only
preliminary setup required is changing the pickling backend to 
[`dill`](https://github.com/uqfoundation/dill)
```python
import os
os.environ["JULIA_PYTHONCALL_PICKLE"] = "dill"
```
This is needed because PythonCall uses pickling for serialization, a process
necessary for distributed sampling. But the default `pickle` module cannot handle
complex functions. With this, running PT on multiple processes is as simple as  
```python
result = jl.pigeons(
    target=path,
    n_chains=3,
    record = recorders,
    checkpoint = True,
    on = jl.Pigeons.ChildProcess(
        n_local_mpi_processes=3,
        dependencies=[jl.PythonCall,jl.NumPygeons]
    )
)
pt = jl.Pigeons.load(result) # load the PT object from the checkpoint
jl.get_sample(pt) # get samples, etc...
```
A key part of this code working is our inclusion of `import` statements 
**inside the model definition**. This is necessary whenever your NumPyro model
is defined in the global python module, as we have done here. If we didn't 
include these statements in the model, we would encounter these errors
```python
ERROR: Python: NameError: name 'numpyro' is not defined
```
On the other hand, if your model is **defined inside a proper python package** that 
itself imports NumPyro, and which exists in the same python virtual environment, 
then it is not necessary to include the import statements inside the model
definition. Everything will just work.


## Limitations

- Only supports the MCMC samplers in the 
[**autostep** package](https://github.com/UBC-Stat-ML/autostep).
- For this reason, NumPygeons only supports models where all the latent variables
are **continuous**.
- Asymptotically, the package suffers from no overhead. **However**, there is a 
significant per-round fixed cost associated with recompiling the adapted tempered
potentials at the beginning of the round. Specifically, `n_chains` tempered 
versions of the target potential function need to be recompiled. For complex models,
each compilation may take up to tens of seconds.
- In multiprocess settings, recompilation needs to be carried out by all processes. 
In theory, this could be prevented by using 
[JAX's persistent compilation cache](https://docs.jax.dev/en/latest/persistent_compilation_cache.html).
Sadly, race conditions seem to arise that result in crashes when the cache is activated. 
- Parallelism invariance **cannot be currently guaranteed**. The cause is still unclear,
but it most likely has to do with the way the `jax.random` module interacts with MPI.
Nevertheless, the results seem to be reproducible when the number of processes
is kept fixed.
- **Multithreaded** sampling is not supported, even when the guidelines
provided in [PythonCall docs](https://juliapy.github.io/PythonCall.jl/dev/pythoncall/#jl-multi-threading)
are followed. This is because the issue is related to race-conditions during JAX tracing.


## Ideas for improvement

- Write the `Pigeons.explore!` loop in terms of a `vmap` across replicas. Should
give a massive speedup even on CPU.

## References

Biron-Lattes, M., Surjanovic, N., Syed, S., Campbell, T., & Bouchard-Côté, A.. (2024). 
[autoMALA: Locally adaptive Metropolis-adjusted Langevin algorithm](https://proceedings.mlr.press/v238/biron-lattes24a.html). 
*Proceedings of The 27th International Conference on Artificial Intelligence and Statistics*, 
in *Proceedings of Machine Learning Research* 238:4600-4608.

Liu, T., Surjanovic, N., Biron-Lattes, M., Bouchard-Côté, A., & Campbell, T. (2024). 
[AutoStep: Locally adaptive involutive MCMC](https://arxiv.org/abs/2410.18929). arXiv preprint arXiv:2410.18929.

Surjanovic, N., Biron-Lattes, M., Tiede, P., Syed, S., Campbell, T., Bouchard-Côté, A.. (2025).
[Pigeons.jl: Distributed sampling from intractable distributions](https://doi.org/10.21105/jcon.00139).
*The Proceedings of the JuliaCon Conferences*, 7(69), 139,