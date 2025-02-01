import catalax as ctx
import numpy as np
import jax.numpy as jnp
import json
import brainunit as u
import brainstate as bst
import catalax.mcmc as cmc

bst.environ.set(precision=64)


# Initialize the model
model = ctx.Model(name="Simple menten model")

def f(t, y, args):
    K_m, v_max  = args
    s1 = y * (u.molar / u.katal)
    d_s1 = - (v_max * s1) / (K_m + s1)
    return d_s1 / u.second

# Add term
model.add_term(f)

# Add species
model.add_species("s1")

# Add ODEs
model.add_ode("s1", "- (v_max * s1) / ( K_m + s1)")

# Prepare the model for bayes and define priors 
model.parameters.v_max.value = 7.0 * u.katal
model.parameters.K_m.value = 100.0 * u.molar
# model.parameters.v_max.value = 7.0
# model.parameters.K_m.value = 100.0


# Data is sampled at different time points
# and also not at zero to reflect realistic
# scenarios
n_ds = 30

time = jnp.array([
    *[[10, 30 ,50 ,70 ,90, 100],
    [15, 35, 55, 78, 98, 108],
    [11, 23, 41 , 68, 86, 110],
    [23, 41, 68, 86, 110, 120],]*n_ds
]) * u.second


# time = jnp.array([
#     *[[10, 30 ,50 ,70 ,90, 100],
#     [15, 35, 55, 78, 98, 108],
#     [11, 23, 41 , 68, 86, 110],
#     [23, 41, 68, 86, 110, 120],]*n_ds
# ])

# Set initial conditions above and below the 
# true Km value for the sake of the example

initial_conditions = []

for _ in range(n_ds):
    initial_conditions += [
        {"s1": np.random.normal(300.0, 8.0) * u.katal},
        {"s1": np.random.normal(200.0, 8.0) * u.katal},
        {"s1": np.random.normal(80.0, 8.0) * u.katal},
        {"s1": np.random.normal(50.0, 8.0) * u.katal},
    ]


# initial_conditions = []
# for _ in range(n_ds):
#     initial_conditions += [
#         {"s1": np.random.normal(300.0, 8.0)},
#         {"s1": np.random.normal(200.0, 8.0)},
#         {"s1": np.random.normal(80.0, 8.0)},
#         {"s1": np.random.normal(50.0, 8.0)},
#     ]

time, data = model.simulate(
    initial_conditions=initial_conditions,
    dt0=0.1 * u.second, saveat=time, in_axes=(0, None, 0)
)

# time, data = model.simulate(
#     initial_conditions=initial_conditions,
#     dt0=0.1, saveat=time, in_axes=(0, None, 0)
# )

# Add some noise for realism
data = np.random.normal(data.to_decimal(u.katal), 5.0).clip(min=0) * u.katal

# Add some noise for realism
# data = np.random.normal(data, 5.0).clip(min=0)

# Turn intiial conditions into a matrix (Not yet part of the NeuralODE workflow)
y0s = model._assemble_y0_array(initial_conditions, in_axes=(0, None, None))



print(f"Time: {time.shape} | Data: {data.shape} | Initial Conditions: {y0s.shape}")

# Define Priors
model.parameters.v_max.prior = cmc.priors.Uniform(low=1e-6, high=200.0, unit=u.katal)
model.parameters.K_m.prior = cmc.priors.Uniform(low=1e-6, high=1e3, unit=u.molar)
# model.parameters.v_max.prior = cmc.priors.Uniform(low=1e-6, high=200.0)
# model.parameters.K_m.prior = cmc.priors.Uniform(low=1e-6, high=1e3)



# Perform MCMC simulation
mcmc, bayes_model = cmc.run_mcmc(
    model=model,
    data=data,
    initial_conditions=initial_conditions,
    times=time,
    yerrs=2.0,
    num_warmup=20,
    num_samples=20,
    dt0=0.1 * u.second,
    max_steps=64**4
)

# Add parameters to the model
for param, samples in mcmc.get_samples().items():
    if param not in model.parameters:
        continue

    model.parameters[param].value = float(samples.mean()) * model.parameters[param].prior.unit

print(model.parameters)

# # Visualize the result using a corner plot
# # Shows the posterior distribution of the parameters
# # Shows the correlation between the parameters
# fig = cmc.plot_corner(mcmc, model=model)

f = ctx.visualize(
    model=model,
    data=data[:4],
    times=time[:4],
    initial_conditions=initial_conditions[:4],
    figsize=(4,2),
    mcmc=mcmc
)