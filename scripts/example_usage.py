# prevent PythonCall+juliacall from using their own environments
import os
os.environ["JULIA_CONDAPKG_BACKEND"] = "NULL"
os.environ["PYTHON_JULIAPKG_PROJECT"] = os.getcwd()
os.environ["PYTHON_JULIAPKG_OFFLINE"] = "yes"

# import requirements
from juliacall import Main as jl
jl.seval("using Pigeons, NumPygeons")
from numpygeons.toy_examples import toy_unid_example
from autostep.autohmc import AutoMALA
from autostep.selectors import AsymmetricSelector

# load the toy model
model, model_args, model_kwargs = toy_unid_example()

# setup the Pigeons target
kernel_type = AutoMALA
kernel_kwargs = {'selector': AsymmetricSelector()}
path = jl.NumPygeons.NumPyroPath(
    model=model,
    model_args=model_args,
    model_kwargs=model_kwargs,
    kernel_type=kernel_type,
    kernel_kwargs=kernel_kwargs
    )

# run single process
pt = jl.Pigeons.pigeons(target = path, n_chains = 4)

# run multiprocess via ChildProcess
pt = jl.Pigeons.pigeons(
    target = path, 
    n_chains = 4,
    on = jl.Pigeons.ChildProcess(
        n_local_mpi_processes=4,
        dependencies=[jl.PythonCall,jl.NumPygeons]
    )
)
