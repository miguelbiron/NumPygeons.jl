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
import jax
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

Having defined the model, running Pigeons is easy. We will use
[AutoMALA](https://proceedings.mlr.press/v238/biron-lattes24a.html) 
from the [autostep package](https://github.com/UBC-Stat-ML/autostep)
as the MCMC kernel. 
```python
from autostep.autohmc import AutoMALA

path = jl.NumPyroPath(
    mcmc_kernel = AutoMALA(eight_schools),
    model_args = (sigma,),
    model_kwargs={'y': y},
)
pt = jl.pigeons(target=path, n_chains=3)

[ Info: `model_args` was a Julia Tuple; converting to python tuple
[ Info: `model_kwargs` was a PythonCall.PyDict; converting to python dict
┌ Info: Neither traces, disk, nor online recorders included.
│    You may not have access to your samples (unless you are using a custom recorder, or maybe you just want log(Z)).
└    To add recorders, use e.g. pigeons(target = ..., record = [traces; record_default()])
────────────────────────────────────────────────────────────────────────────
  scans        Λ        time(s)    allc(B)  log(Z₁/Z₀)   min(α)     mean(α) 
────────── ────────── ────────── ────────── ────────── ────────── ──────────
        2          0       4.03   4.29e+04      -31.9          1          1 
        4      0.698       1.58   2.12e+04      -32.3      0.302      0.651 
        8      0.619     0.0233   3.77e+04      -30.8      0.403      0.691 
       16      0.861      0.043   7.17e+04      -31.3      0.517      0.569 
       32      0.808     0.0763   1.39e+05      -31.8      0.506      0.596 
       64      0.941      0.155   2.73e+05      -31.7      0.451      0.529 
      128      0.929      0.333   5.41e+05      -31.1      0.512      0.535 
      256      0.889        0.6   1.08e+06        -31      0.488      0.555 
      512      0.956       1.17   2.15e+06      -31.4      0.481      0.522 
 1.02e+03      0.929       2.44    4.3e+06      -31.2      0.532      0.535 
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
        2          0          0     0.0704   1.24e+06      -31.9          1          1          1          1 
        4          0      0.698      0.017   3.68e+04      -32.3      0.302      0.651      0.998      0.771 
        8          1      0.619     0.0291   6.73e+04      -30.8      0.403      0.691      0.974      0.891 
       16          2      0.861     0.0528   1.26e+05      -31.3      0.517      0.569      0.993      0.993 
       32          6      0.808     0.0976    2.4e+05      -31.8      0.506      0.596      0.955      0.945 
       64          7      0.941      0.199   4.68e+05      -31.7      0.451      0.529      0.972      0.919 
      128         22      0.929      0.404   9.24e+05      -31.1      0.512      0.535      0.976      0.956 
      256         38      0.889      0.794   1.84e+06        -31      0.488      0.555      0.981      0.975 
      512         71      0.956       1.58   3.66e+06      -31.4      0.481      0.522      0.987      0.979 
 1.02e+03        152      0.929        3.3   7.31e+06      -31.2      0.532      0.535      0.985      0.982 
─────────────────────────────────────────────────────────────────────────────────────────────────────────────
```

We can then for example inspect the data using arviz
```python
import arviz as az

# add a singleton "chain" dimension to conform to arviz specification
samples = jl.get_sample(pt)
az_samples=az.from_dict(
  jax.tree.map(lambda x: jnp.expand_dims(x, axis=0), samples)
)
az.summary(az_samples,var_names=["mu","tau","theta"])

arviz - WARNING - Shape validation failed: input_shape: (1, 1024), minimum_shape: (chains=2, draws=4)
           mean     sd  hdi_3%  hdi_97%  mcse_mean  mcse_sd  ess_bulk  ess_tail  r_hat
mu        4.352  3.399  -2.158   10.273      0.371    0.211      85.0     102.0    NaN
tau       3.728  2.825   0.015    9.193      0.306    0.246      94.0     187.0    NaN
theta[0]  6.224  4.804  -2.017   15.290      0.469    0.293     107.0     108.0    NaN
theta[1]  4.537  4.209  -3.037   11.852      0.398    0.187     109.0     223.0    NaN
theta[2]  3.972  5.342  -4.507   16.165      0.556    0.428      92.0      74.0    NaN
theta[3]  5.210  5.050  -3.570   14.814      0.578    0.431      73.0     100.0    NaN
theta[4]  3.686  4.706  -5.442   12.877      0.507    0.361      87.0     125.0    NaN
theta[5]  4.112  4.653  -4.941   13.323      0.422    0.363     120.0     150.0    NaN
theta[6]  6.733  4.610  -2.111   14.720      0.464    0.253     101.0     209.0    NaN
theta[7]  4.877  5.298  -4.529   14.364      0.521    0.373     102.0     201.0    NaN
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