import json

import numpy as np
import jax.numpy as jnp
import catalax as ctx
import catalax.neural as ctn

import brainunit as u

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

time, data = model.simulate(
    initial_conditions=initial_conditions,
    dt0=0.1 * u.second, saveat=time, in_axes=(0, None, 0)
)

# Add some noise for realism
data = np.random.normal(data.to_decimal(u.katal), 1.0).clip(min=0) * u.katal

# Turn intiial conditions into a matrix (Not yet part of the NeuralODE workflow)
y0s = model._assemble_y0_array(initial_conditions, in_axes=(0, None, None))


print(f"Time: {time.shape} | Data: {data.shape} | Initial Conditions: {y0s.shape}")

# Create a neural ODE model
rbf = ctn.RBFLayer(0.6)
neural_ode = ctn.NeuralODE.from_model(model, width_size=16, depth=1, activation=rbf)

# Set up a training strategy (You can interchange the loss function too!)
strategy = ctn.Strategy()
strategy.add_step(lr=1e-3, length=0.1, steps=1000, batch_size=20, alpha=0.1)
strategy.add_step(lr=1e-3, steps=3000, batch_size=20, alpha=0.01)
strategy.add_step(lr=1e-4, steps=5000, batch_size=20, alpha=0.01)

# Train neural ODE
trained = ctn.train_neural_ode(
    model=neural_ode,
    data=data,
    times=time,
    inital_conditions=y0s,
    strategy=strategy,
    sigma=0.03,
    n_augmentations=10,
    print_every=10,
    weight_scale=1e-3,
    save_milestones=False, # Set to True to save model checkpoints
    # log="progress.log", # Uncomment this line to log progress
)
