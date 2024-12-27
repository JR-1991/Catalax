<h1 align='center'>CatalaxU</h1>
<h2 align='center'>Unit-aware biological systems modeling in JAX.</h2>

Catalax is a JAX-based framework that facilitates [unit-aware](https://github.com/chaobrain/brainunit) simulation and parameter inference through optimization algorithms and Hamiltonian Monte Carlo sampling. Its features enable efficient model building and inference, including the utilization of neural ODEs to model system dynamics and serve as surrogates for the aforementioned techniques.

🚧 Please note that Catalax is still in early development and the API is subject to change. 🚧

## Getting started

To get started with Catalax, you can install it via pip

```bash
pip install git+https://github.com/routhleck/catalax.git
pip install git+https://github.com/chaoming0625/diffrax.git
```
or by source

```bash
git clone https://github.com/routhleck/Catalax.git
cd Catalax
pip install .
pip install -r requirements.txt
```

## Quickstart

To develop a model, Catalax offers a user-friendly interface that comprises two core components: `Species` and `ODE`. The former is utilized to specify the species of the model, while the latter is used to define its dynamics. Through the integration of these components, a robust model is created, which can be employed for inference purposes. Notably, Catalax automatically generates `Parameter` objects from the extracted parameters, which can be leveraged to define priors and constraints for the model.

```python
import catalax as ctx
import brainunit as u


m# Initialize the model
model = ctx.Model(name="Simple Menten model")

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

# Turn initial conditions into a matrix (Not yet part of the NeuralODE workflow)
y0s = model._assemble_y0_array(initial_conditions, in_axes=(0, None, None))

# Visualize the data
f = ctx.visualize(
    model=model,
    data=data[:4],
    times=time[:4],
    initial_conditions=initial_conditions[:4],
    figsize=(4,2),
)
```

More examples for unit-aware numerical integration please see [unit-aware examples](https://github.com/routhleck/catalax/blob/main/examples-with-units).