import catalax as ctx
import catalax.neural as ctn
import equinox as eqx
from jax.flatten_util import ravel_pytree

model = ctx.Model(name="Michaelis-Menten")
model.add_species(s1="Substrate")
model.add_constant(e="Enzyme")
model.add_ode("s1", "kcat * e * s1 / (k_m + s1)")

neural_ode = ctn.NeuralODE.from_model(model, width_size=6, depth=2)

params, static = eqx.partition(neural_ode, eqx.is_array)
theta0, unravel = ravel_pytree(params)

print(unravel)
