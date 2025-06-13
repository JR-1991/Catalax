import catalax as ctx
import catalax.mcmc as cmc
import jax.numpy as jnp

# Define the model
model = ctx.Model(name="TestModel")

model.add_species(s1="Substrate")
model.add_constant(e="Enzyme")
model.add_ode("s1", "- kcat * e * s1 / ( Km + s1 )")

model.parameters.Km.value = 100.0
model.parameters.Km.prior = cmc.priors.Uniform(low=30.0, high=180.0)
model.parameters.kcat.value = 3.0
model.parameters.kcat.prior = cmc.priors.Uniform(low=0.1, high=10.0)

# Create a dataset
dataset = ctx.Dataset.from_model(model)
dataset.add_initial(s1=100.0, e=0.5)
dataset.add_initial(s1=100.0, e=1.0)
dataset.add_initial(s1=100.0, e=2.0)

config = ctx.SimulationConfig(
    nsteps=10,
    t0=jnp.array([0, 10, 20]),
    t1=jnp.array([10, 30, 40]),
)

# Simulate the model
dataset = model.simulate(
    dataset=dataset,
    config=config,
)

model.parameters.Km.value = 10.0
model.parameters.Km.prior = cmc.priors.Uniform(low=30.0, high=180.0)
model.parameters.kcat.value = 3.0
model.parameters.kcat.prior = cmc.priors.Uniform(low=0.1, high=10.0)

dataset.plot(
    show=False,
    path="test.png",
    model=model,
)
