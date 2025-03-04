#!/usr/bin/env bash

export JULIA_CONDAPKG_BACKEND="Null"
export PYTHON_JULIAPKG_PROJECT="Project.toml"
export PYTHON_JULIAPKG_OFFLINE="yes"

if command -v pip; then
    echo found pip
else
    echo pip not found
    exit 1
fi

if command -v julia; then
    echo found julia
else
    echo julia not found
    exit 1
fi

# upgrade pip
pip install --upgrade pip

# install juliacall
pip install juliacall

# install julia dependencies
jl_instructions=$(cat <<-END
using Pkg
Pkg.add([
    (;name="PythonCall"),
    (;name="Pigeons", rev="main"),
    (;path="/home/mbiron/projects/NumPygeons.jl")
])
END

)
julia --project=${PYTHON_JULIAPKG_PROJECT} -e "${jl_instructions}"

# install numpygeons module
jl_instructions=$(cat <<-END
println(joinpath(dirname(dirname(Base.find_package("NumPygeons"))), "numpygeons"))
END

)
numpygeonsdir=$(julia --project=${PYTHON_JULIAPKG_PROJECT} -e "${jl_instructions}")
pip install ${numpygeonsdir}

