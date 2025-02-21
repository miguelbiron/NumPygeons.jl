# input validation
is_python_tuple(t) = (t isa Py && pyisinstance(t, pytype(pytuple(()))))
is_python_dict(d) = (d isa Py && pyisinstance(d, pytype(pydict())))
