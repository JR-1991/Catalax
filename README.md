
<div align="center">
<h1>Catalax 🧬⚡</h1>

**A High-Performance JAX Framework for Biochemical Modeling, Neural ODEs, and Bayesian Inference**

[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://python.org)
[![JAX](https://img.shields.io/badge/JAX-powered-orange.svg)](https://github.com/google/jax)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

*Accelerate your biochemical modeling with differentiable programming*

</div>

## 🚀 What is Catalax?

Catalax combines the power of **JAX** with advanced numerical methods for biochemical modeling. It seamlessly integrates traditional mechanistic modeling with cutting-edge neural differential equations, enabling:

- **⚡ Lightning-fast simulations** with JIT compilation and GPU acceleration
- **🧠 Neural ODEs** for data-driven discovery of biochemical dynamics  
- **🎯 Bayesian parameter inference** using Hamiltonian Monte Carlo
- **📊 Comprehensive model analysis** with fit metrics and model comparison
- **🔬 Biochemical data integration** with EnzymeML support

🚧 **Note**: Catalax is in active development. API may change in future versions.

## 🌟 Key Features

### 🔬 **Mechanistic Modeling**

- Intuitive model building with species, reactions, and ODEs
- Automatic parameter extraction and constraint handling
- Support for complex biochemical networks and pathways
- Integration with experimental data and EnzymeML format

### 🧠 **Neural Differential Equations**

- **NeuralODE**: Pure neural network dynamics learning
- **RateFlowODE**: Reaction network discovery with learnable stoichiometry
- **UniversalODE**: Hybrid models combining mechanistic and neural components
- Advanced penalty systems for biological constraints

### 🎯 **Bayesian Inference**

- MCMC sampling with NUTS (No-U-Turn Sampler)
- Surrogate-accelerated inference using neural networks
- Uncertainty quantification with HDI intervals
- Prior specification and model comparison tools

### 📊 **Model Analysis**

- Statistical model comparison (AIC, BIC, chi-square)
- Phase plot visualization for rate analysis
- Comprehensive plotting and visualization tools (phase plots, rate grids, etc.)

## 🛠️ Installation

```bash
# Install from PyPI
pip install catalax

# Or from source
git clone https://github.com/JR-1991/Catalax.git
cd Catalax
pip install .
```

## 🚀 Quick Start

### 1. **Mechanistic Modeling**

Create biochemical models using first-principles knowledge of reaction mechanisms. This approach is ideal when you understand the underlying biochemistry and want to build physically interpretable models. Catalax automatically extracts parameters from equations and enables efficient simulation with JAX compilation.

**Key features:**

- **Automatic parameter extraction** from symbolic equations
- **Multiple initial conditions** for batch simulation
- **Fast JAX-compiled** integration with adaptive step sizes
- **Built-in visualization** with publication-ready plots

<details open>
<summary><strong>📖 Click to see mechanistic modeling code</strong></summary>

```python
import catalax as ctx

# Create a Michaelis-Menten enzyme kinetics model
model = ctx.Model(name="Enzyme Kinetics")

# Define species
model.add_species(
    S="Substrate",
    E="Enzyme", 
    ES="Enzyme-Substrate Complex",
    P="Product"
)

# Add reaction kinetics
model.add_ode("S", "-k1*E*S + k2*ES")
model.add_ode("E", "-k1*E*S + k2*ES + k3*ES")
model.add_ode("ES", "k1*E*S - k2*ES - k3*ES")
model.add_ode("P", "k3*ES")

# Set parameters
model.parameters.k1.value = 0.1
model.parameters.k2.value = 0.05
model.parameters.k3.value = 0.02

# Create dataset and add initial conditions
dataset = ctx.Dataset.from_model(model)
dataset.add_initial(S=100, E=10, ES=0, P=0)
dataset.add_initial(S=200, E=10, ES=0, P=0)  # Different initial conditions

# Configure simulation
config = ctx.SimulationConfig(t0=0, t1=100, nsteps=1000)

# Run simulation
results = model.simulate(dataset=dataset, config=config)

# Visualize results
results.plot()
```

</details>

### 2. **Neural ODEs for Discovery**

Discover unknown reaction networks directly from experimental time-series data using neural differential equations. The `RateFlowODE` learns both reaction rates and stoichiometric coefficients, automatically uncovering the underlying biochemical network structure without prior mechanistic knowledge.

**Key features:**

- **Learnable stoichiometry** matrices that discover reaction networks
- **Biological constraints** through penalty functions (mass balance, sparsity, integer coefficients)
- **Multi-step training** strategies with adaptive learning rates
- **Rate visualization** to interpret discovered reactions

<details open>
<summary><strong>🧠 Click to see neural ODE discovery code</strong></summary>

```python
from catalax.neural import RateFlowODE, train_neural_ode, Strategy

# Create experimental dataset (with measurement data)
experimental_data = ctx.Dataset.from_model(model)
experimental_data.add_initial(S=100, E=10, ES=0, P=0)
# Add actual measurement data to the dataset...

# Create a neural ODE that learns reaction stoichiometry
neural_model = RateFlowODE.from_dataset(
    dataset=experimental_data,
    reaction_size=3,  # Number of reactions to discover
    width_size=64,
    depth=3
)

# Set up training strategy
strategy = Strategy()
strategy.add_step(lr=1e-3, steps=1000, batch_size=32)
strategy.add_step(lr=1e-4, steps=2000, batch_size=64)

# Train the model
trained_model = train_neural_ode(
    model=neural_model,
    dataset=experimental_data,
    strategy=strategy
)

# Visualize learned reactions
trained_model.plot_learned_rates(experimental_data, original_model)
```

</details>

### 3. **Bayesian Parameter Inference**

Quantify parameter uncertainty and obtain credible intervals using Hamiltonian Monte Carlo (HMC) sampling. This approach goes beyond point estimates to provide full posterior distributions, enabling robust uncertainty quantification and model comparison through Bayesian statistics.

**Key features:**

- **NUTS sampler** (No-U-Turn Sampler) for efficient exploration of parameter space
- **Multiple chains** for convergence diagnostics and robust sampling
- **Flexible priors** supporting various probability distributions
- **HDI intervals** (Highest Density Intervals) for credible parameter ranges
- **Integrated visualization** with corner plots, trace plots, and credibility intervals

<details>
<summary><strong>🎯 Click to see Bayesian inference code</strong></summary>

```python
from catalax.mcmc import HMC
from catalax.mcmc.priors import Normal, LogNormal
import numpyro.distributions as dist

# Set parameter priors
model.parameters.k1.prior = LogNormal(mu=0.0, sigma=1.0)
model.parameters.k2.prior = LogNormal(mu=0.0, sigma=1.0)
model.parameters.k3.prior = LogNormal(mu=0.0, sigma=1.0)

# Create HMC sampler with configuration
hmc = HMC(
    num_warmup=1000,
    num_samples=2000,
    likelihood=dist.SoftLaplace,  # or dist.Normal
    num_chains=4,
    chain_method="parallel"
)

# Create experimental dataset
experimental_data = ctx.Dataset.from_model(model)
experimental_data.add_initial(S=100, E=10, ES=0, P=0)
# Add more experimental measurements as needed...

# Run inference
results = hmc.run(
    model=model,
    dataset=experimental_data,
    yerrs=0.1  # Measurement uncertainty
)

# Analyze results with integrated visualization
samples = results.get_samples()
results.print_summary()

# Create publication-ready plots
fig1 = results.plot_corner()
fig2 = results.plot_posterior()
fig3 = results.plot_trace()
fig4 = results.plot_credibility_interval(
    initial_condition={"S": 100, "E": 10, "ES": 0, "P": 0},
    time=jnp.linspace(0, 100, 200)
)

# Get summary statistics
summary_stats = results.summary(hdi_prob=0.95)
```

</details>

### 4. **Advanced Neural Network Training**

Train neural ODEs with sophisticated biological constraints and multi-stage optimization strategies. The penalty system enforces biochemical principles like mass conservation, reaction sparsity, and integer stoichiometry, while the training strategy allows fine-grained control over different model components.

**Key features:**

- **Biological penalties** that enforce mass balance, sparsity, and integer coefficients
- **Multi-stage training** with different learning rates and penalty weights
- **Component-specific training** (MLP only, stoichiometry only, or both)
- **Adaptive penalty scheduling** for improved convergence
- **Built-in regularization** with L1/L2 weight penalties

<details>
<summary><strong>⚙️ Click to see advanced training code</strong></summary>

```python
from catalax.neural.penalties import Penalties

# Create penalty system for biological constraints
penalties = Penalties.for_rateflow(
    alpha=0.1,
    density_alpha=0.05,      # Encourage sparse reactions
    bipolar_alpha=0.1,       # Enforce mass balance
    integer_alpha=0.02,      # Encourage integer stoichiometry
    l2_alpha=0.01           # L2 regularization
)

# Create training dataset
training_dataset = ctx.Dataset.from_model(model)
training_dataset.add_initial(S=100, E=10, ES=0, P=0)

# Multi-step training with different penalties
strategy = Strategy()
strategy.add_step(
    lr=1e-3, steps=1000, batch_size=32,
    penalties=penalties,
    train=ctx.neural.Modes.MLP  # Train only neural network
)
strategy.add_step(
    lr=1e-4, steps=1000, batch_size=64,
    penalties=penalties.update_alpha(0.05),
    train=ctx.neural.Modes.BOTH  # Train network + stoichiometry
)

trained_model = train_neural_ode(neural_model, training_dataset, strategy)
```

</details>

## 🎨 Advanced Features

### 🔬 **EnzymeML Integration**

Seamlessly integrate with the EnzymeML standard for biochemical data exchange. Import experimental data and model structures from EnzymeML documents, run analysis in Catalax, and export results back to the standardized format for interoperability with other tools.

<details>
<summary><strong>🔬 Click to see EnzymeML integration code</strong></summary>

```python
import pyenzyme as pe

# Load from EnzymeML
doc = pe.read_enzymeml("experiment.json")
model = ctx.Model.from_enzymeml(doc, from_reactions=True)

# Export back to EnzymeML
model.update_enzymeml_parameters(doc)
```

</details>

### 📊 **Model Comparison**

Evaluate and compare model performance using comprehensive statistical metrics following the lmfit convention. Calculate goodness-of-fit measures and information criteria to assess model quality and select the best model among competing hypotheses.

<details>
<summary><strong>📊 Click to see model comparison code</strong></summary>

```python
from catalax.dataset.metrics import FitMetrics

# Create experimental dataset for comparison
experimental_data = ctx.Dataset.from_model(model)
experimental_data.add_initial(S=100, E=10, ES=0, P=0)

# Calculate metrics using the dataset's built-in method
metrics = experimental_data.metrics(predictor=model)

print(f"AIC: {metrics.aic:.2f}")
print(f"BIC: {metrics.bic:.2f}")
print(f"Reduced χ²: {metrics.redchi:.3f}")
```

</details>

### 🎯 **Surrogate-Accelerated MCMC**

Dramatically accelerate Bayesian inference by replacing expensive mechanistic model evaluations with fast neural network surrogates. Train a neural ODE on synthetic data, then use it as a proxy during MCMC sampling to achieve 10-100x speedups while maintaining accuracy.

<details>
<summary><strong>🎯 Click to see surrogate MCMC code</strong></summary>

```python
from catalax.mcmc import HMC
from catalax.neural import train_neural_ode

# Create dataset for surrogate training
training_data = ctx.Dataset.from_model(mechanistic_model)
training_data.add_initial(S=100, E=10, ES=0, P=0)

# Train neural surrogate
surrogate = train_neural_ode(neural_model, training_data, strategy)

# Create HMC sampler
hmc = HMC(
    num_warmup=1000,
    num_samples=2000,
    num_chains=4,
    chain_method="parallel"
)

# Use surrogate for fast MCMC (10-100x speedup!)
results = hmc.run(
    model=mechanistic_model,
    dataset=experimental_data,
    yerrs=0.1,
    surrogate=surrogate
)

# Analyze results
results.print_summary()
fig = results.plot_corner()
credible_intervals = results.plot_credibility_interval(
    initial_condition={"S": 100, "E": 10, "ES": 0, "P": 0},
    time=jnp.linspace(0, 100, 200)
)
```

</details>

### 📈 **Phase Plot Analysis**

Visualize how reaction rates depend on species concentrations using interactive phase plots and heatmaps. This powerful analysis tool helps understand rate limiting steps, identify optimal operating conditions, and reveal complex concentration dependencies in biochemical networks.

<details>
<summary><strong>📈 Click to see phase plot analysis code</strong></summary>

```python
# Create dataset for phase plot analysis
analysis_data = ctx.Dataset.from_model(mechanistic_model)
analysis_data.add_initial(S=100, E=10, ES=0, P=0)

# Analyze rate dependencies
rateflow_model.plot_rate_grid(
    dataset=analysis_data,
    model=mechanistic_model,
    species_pairs=[("S", "E"), ("ES", "P")],
    grid_resolution=50
)
```

</details>

## 🔧 Core Components

| Component     | Description                                                       |
| ------------- | ----------------------------------------------------------------- |
| **Model**     | Core mechanistic modeling with ODEs, species, and parameters      |
| **Dataset**   | Experimental data handling with simulation integration            |
| **Neural**    | Neural ODE implementations (NeuralODE, RateFlowODE, UniversalODE) |
| **MCMC**      | Bayesian inference with NumPyro backend                           |
| **Penalties** | Biological constraint enforcement for neural networks             |
| **Metrics**   | Statistical model evaluation and comparison                       |

## 📚 Examples

Explore comprehensive examples in the `examples/` directory:

- **[Optimization](examples/Optimization.ipynb)** - Parameter estimation and model fitting
- **[HMC](examples/HMC.ipynb)** - Bayesian parameter inference
- **[NeuralODE](examples/NeuralODE.ipynb)** - Learning dynamics from data
- **[UniversalODE](examples/UniversalODE.ipynb)** - Hybrid mechanistic-neural models
- **[SurrogateHMC](examples/SurrogateHMC.ipynb)** - Accelerated inference with surrogates
- **[Data Import](examples/DataImport.ipynb)** - Working with experimental data

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

Catalax builds on the excellent work of:

- [JAX](https://github.com/google/jax) for high-performance computing
- [NumPyro](https://github.com/pyro-ppl/numpyro) for probabilistic programming
- [Equinox](https://github.com/patrick-kidger/equinox) for neural networks
- [Diffrax](https://github.com/patrick-kidger/diffrax) for differential equations

## 📞 Support

- 💬 [Discussions](https://github.com/JR-1991/Catalax/discussions)
- 🐛 [Issues](https://github.com/JR-1991/Catalax/issues)

---

<div align="center">
<strong>Accelerate your biochemical modeling with Catalax! ⚡</strong>
</div>
