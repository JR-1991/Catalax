# Catalax

Catalax is a JAX-based framework that facilitates simulation and parameter inference through optimization algorithms and Hamiltonian Monte Carlo sampling. Its features enable efficient model building and inference, including the utilization of neural ODEs to model system dynamics and serve as surrogates for the aforementioned techniques.

🚧 Please note that Catalax is still in early development and the API is subject to change. 🚧

## Getting started

To get started with Catalax, you can install it via pip:

**MacOS / Linux**

```bash
python3 -m pip install git+https://github.com/JR-1991/Catalax.git
```

**Windows**

```bash
python -m pip install git+https://github.com/JR-1991/Catalax.git
```

## Quickstart

To develop a model, Catalax offers a user-friendly interface that comprises two core components: `Species` and `ODE`. The former is utilized to specify the species of the model, while the latter is used to define its dynamics. Through the integration of these components, a robust model is created, which can be employed for inference purposes. Notably, Catalax automatically generates `Parameter` objects from the extracted parameters, which can be leveraged to define priors and constraints for the model.

```python
import catalax as ctx

model = ctx.Model(name="My Model")

# Define the species of the model
model.add_species(s1="Substrate", e1="Enzyme")

# Now add an ODE for each species
model.add_ode("s1", "k_cat * e1 * s1 / (K_m + s1)")
model.add_ode("e1", "0", observable=False)

# All parameters [k_cat, K_m] are automatically extracted
# and can be accessed via model.parameters
model.parameters.k_cat.value = 5.0
model.parameters.K_m.value = 100.0

# Integrate over time
initial_condition = {"s1": 100.0, "s2": 0.0}
time, states = model.simulate(
    initial_conditions=initial_condition,
    t0=0, t1=100, dt0=0.1, nsteps=1000, in_axes=None
)

# Visualize the results
f = visualize(
    model=model,
    data=states, # Replace this with actual data
    times=time,
    initial_conditions=initial_conditions,
    figsize=(4,4),
)

```

### Give it a try!

To get a better understanding of Catalax, we recommend that you try out the examples found in the `examples` directory. These examples are designed to showcase the capabilities of Catalax and provide a starting point for your own projects:

* [Basic functions](/examples/Basics.ipynb) - An overview to models and integration
* [Optimization](/examples/Optimization.ipynb) - How to perform parameter estimation using Catalax
* [Non observable species](/examples/Optimization.ipynb) - How to deal with non-observable species
* [Hamiltonian MC](/examples/HMC.ipynb) - How to perform parameter inference using Hamiltonian Monte Carlo
* [Neural Ordinary Differential Equations](/examples/Neural_ODE.ipynb) - How to use neural ODEs to model system dynamics
* [Neural ODEs and HMC](/examples/SurrogateHMC.ipynb) - How to perform parameter inference using surrogate HMC
