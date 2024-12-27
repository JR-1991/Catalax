import numpy as np
import jax.numpy as jnp

import catalax as ctx
import catalax.mcmc as cmc

import brainunit as u

model = ctx.Model(name="Non-Observable Menten Inhibition")

# Define the term function
def f(t, y, args):
    K_m, k_cat, k_d = args
    s1, e1 = y
    d_s1 = - (e1 * k_cat * s1) / (K_m + s1)
    d_e1 = -k_d * e1
    return [d_s1 / u.second, d_e1 / u.second]

# Add term
model.add_term(f)

# Another way of adding species (symbols and names at the same type)
model.add_species(
    s1="Substrate",
    e1="Enzyme",
)

# In this application 'e1' is not observable and thus
# 'k_d' needs to be inferred implicitly from the data.
model.add_ode("s1", "- e1 * k_cat * s1 / (s1 + K_m)")
model.add_ode("e1", "-k_d", observable=False)

model.parameters.K_m.value = 100.0 * u.molar
model.parameters.k_cat.value = 10.0 * u.katal
model.parameters.k_d.value = 0.0001 / u.second

# Create some dummy data to test
initial_conditions = [
    {"s1": 10.0 * u.katal, "e1": 0.1 * u.molar},
    {"s1": 50.0 * u.katal, "e1": 0.1 * u.molar},
    {"s1": 75.0 * u.katal, "e1": 0.1 * u.molar},
    {"s1": 100.0 * u.katal, "e1": 0.1 * u.molar},
    {"s1": 200.0 * u.katal, "e1": 0.1 * u.molar},
    {"s1": 300.0 * u.katal, "e1": 0.1 * u.molar},
    {"s1": 400.0 * u.katal, "e1": 0.1 * u.molar},
]

time, states = model.simulate(
    initial_conditions=initial_conditions,
    dt0=0.1 * u.second, t0=0 * u.second, t1=1000 * u.second, nsteps=10, in_axes=(0, None, None),
)

# Add some noise to the data
print(states)
states = (states + states.to_decimal(u.katal) * np.random.normal(0, 0.02)).clip(min=0)

# Truncate data to only include s1
data = states[:, :, 1]

print(f"Time: {time.shape} | Data: {data.shape}")