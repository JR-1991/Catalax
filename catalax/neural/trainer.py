import os
from datetime import datetime
from typing import Callable, Dict, Optional, Tuple, Type, TypeVar

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
import optax
import tqdm

from catalax.dataset.dataset import Dataset
from catalax.neural.neuralbase import NeuralBase
from catalax.neural.penalties.penalties import Penalties
from catalax.neural.strategy import Step, Strategy

T = TypeVar("T", bound=NeuralBase)
MakeStep = Callable[
    ...,
    Tuple[jax.Array, T, optax.OptState],
]
GradLoss = Callable[
    [T, T, jax.Array, jax.Array, jax.Array, Penalties, jax.Array],
    jax.Array,
]


def train_neural_ode(
    model: T,
    dataset: Dataset,
    strategy: Strategy,
    optimizer=optax.adabelief,
    validation_dataset: Optional[Dataset] = None,
    print_every: int = 100,
    log: Optional[str] = None,
    save_milestones: bool = False,
    milestone_dir: str = "./milestones",
    weight_scale: float = 1e-2,
    solver: Optional[Type[diffrax.AbstractSolver]] = None,
    seed: int = 420,
    progress_bar: bool = True,
    temporal_dropout_p: float = 0.0,
) -> T:
    """Train a neural ODE.

    Args:
        model: NeuralBase model to train.
        dataset: Training dataset.
        strategy: Multi-stage training strategy.
        optimizer: Optax optimizer constructor.
        validation_dataset: Optional validation dataset.
        print_every: Print metrics every N steps.
        log: Optional path to a log file.
        save_milestones: Whether to save per-stage model checkpoints.
        milestone_dir: Directory for milestone checkpoints.
        weight_scale: Initial weight rescaling factor.
        solver: Optional diffrax solver override.
        seed: PRNG seed.
        progress_bar: Whether to show a tqdm progress bar.
        temporal_dropout_p: Probability of *dropping* an interior time point
            from the loss at each step. The initial condition (t=0) is never
            dropped. Default is 0.0 (no dropout). A value of 0.2 means each
            interior time point has a 20% chance of being masked out of the
            training loss for that step.

    Returns:
        The trained model.
    """
    # Set whether to print progress bar
    strategy._print = progress_bar

    # Extract data from dataset
    data, times, inital_conditions = dataset.to_jax_arrays(
        state_order=dataset.get_observable_states_order(),
        inits_to_array=True,
    )

    # Set up PRNG keys
    key = jrandom.PRNGKey(seed)
    _, batch_key, loader_key = jrandom.split(key, 3)
    _, length_size, _ = data.shape

    # Scale weights and set maximum time
    model = _scale_weights(model, weight_scale)
    model = eqx.tree_at(
        lambda tree: tree.func.max_time,
        model,
        replace=times.max(),
    )

    if solver is not None:
        model = eqx.tree_at(
            lambda tree: tree.solver,
            model,
            replace=solver,
        )

    # Set up logger
    if log is not None:
        log_file = open(log, "w")
        log_file.write("strategy\tstep\tmean_loss\tmae\n")
        log_file.close()

    # Set up milestone directory
    if save_milestones:
        os.makedirs(milestone_dir, exist_ok=True)

    if progress_bar:
        print(f"\n🚀 Training {model.__class__.__name__}...\n")

    for strat_index, strat in enumerate(strategy):
        assert strat.batch_size < data.shape[0], (
            f"Batch size of strategy #{strat_index} ({strat.batch_size}) is larger than the dataset size ({data.shape[0]}). Please reduce the batch size to be < {data.shape[0]}."
        )

        # Partition model to get the differentiable part for optimizer initialization
        diff_model, static_model = strat._partition_model(model)

        # Prepare optimizer per strategy
        optim = optimizer(learning_rate=strat.lr)
        opt_state = optim.init(eqx.filter(diff_model, eqx.is_inexact_array))

        # Prepare step and loss function per strategy
        make_step, grad_loss = _prepare_step_and_loss(strat.loss)

        # Apply training strategy
        max_time = max(int(length_size * strat.length) + 1, 2)

        # Truncate data and times
        _times = times[:, :max_time]
        _data = data[:, :max_time, :]

        # Prepare data generator
        batches = dataloader(
            (_data, inital_conditions, _times),
            strat.batch_size,
            key=loader_key,
        )

        if validation_dataset is not None:
            val_data, val_times, val_inits = validation_dataset.to_jax_arrays(  # type: ignore
                state_order=model.state_order,
                inits_to_array=True,
            )
            val_data = val_data[:, :max_time, :]
            val_times = val_times[:, :max_time]

        # Set up progress bar
        pbar = tqdm.tqdm(
            total=strat.steps,
            desc="╰── startup",
            disable=not progress_bar,
        )

        for step, (yi, y0i, ti) in zip(range(strat.steps), batches):
            # Generate new keys for each batch
            batch_key = jrandom.fold_in(batch_key, step)
            mask_key, batch_key = jrandom.split(batch_key)

            # Build per-step temporal dropout mask. Shared across the batch
            # so all trajectories see the same dropped time indices, which
            # enforces consistency. The IC (t=0) is always kept.
            mask = temporal_dropout_mask(
                mask_key,
                n_times=ti.shape[1],
                temporal_dropout_p=temporal_dropout_p,
            )

            loss, model, opt_state = make_step(
                ti=ti,
                yi=yi,
                y0i=y0i,
                model=model,
                opt_state=opt_state,
                optimizer=optim,
                partitioned_model=strat._partition_model(model),
                penalties=strat.penalties,
                mask=mask,
            )

            if (step % print_every) == 0 or step == strat.steps - 1:
                # Calculate mean loss over data (no dropout for metrics)
                loss, mae = _calculate_metrics(
                    strat,
                    model,
                    _times,
                    _data,
                    inital_conditions,  # type: ignore
                    grad_loss,
                )

                if validation_dataset is not None:
                    val_loss, val_mae = _calculate_metrics(
                        strat,
                        model,
                        val_times,
                        val_data,
                        val_inits,  # type: ignore
                        grad_loss,
                    )
                    pbar.set_description(
                        f"╰── loss: [{loss:.4f}|{val_loss:.4f}] mae: [{mae:.4f}|{val_mae:.4f}]"
                    )
                else:
                    pbar.set_description(f"╰── loss: {loss:.4f} mae: {mae:.4f}")

                # Update progress bar
                pbar.update(print_every)

                _log_progress(strat_index, step, loss, mae, log)

        pbar.close()

        if progress_bar:
            print("\n")

        if save_milestones:
            # Save milestone
            _serialize_milestone(
                model,
                milestone_dir,
                strat_index,
                {"lr": strat.lr, "steps": strat.steps, "length": strat.length},
            )

    return model


def temporal_dropout_mask(
    key: jax.Array,
    n_times: int,
    temporal_dropout_p: float,
) -> jax.Array:
    """Build a Bernoulli mask over interior time points; t=0 is always kept.

    Semantics match standard dropout: ``temporal_dropout_p`` is the probability
    of *dropping* a point. ``temporal_dropout_p=0.0`` keeps everything (no
    dropout); ``temporal_dropout_p=1.0`` would drop every interior point.

    Args:
        key: PRNG key.
        n_times: Number of time points in the trajectory (including the IC).
        temporal_dropout_p: Probability of dropping each interior time point.

    Returns:
        Float mask of shape ``(n_times,)`` with ``mask[0] == 1.0`` always,
        and ``mask[1:]`` independently 1.0 with probability ``1 -
        temporal_dropout_p`` else 0.0.
    """
    p_keep = 1.0 - temporal_dropout_p
    interior = jrandom.bernoulli(key, p=p_keep, shape=(n_times - 1,))
    interior = interior.astype(jnp.float32)
    return jnp.concatenate([jnp.ones((1,), dtype=jnp.float32), interior])


def _prepare_step_and_loss(loss) -> Tuple[Callable, Callable]:
    """Based on a strategy, prepares the step and loss function.

    This function is meant to prepare the step and loss function based on a
    strategy. The strategy is defined by the loss function that is used for
    training. The loss function can be any type of loss function that is
    compatible with JAX. The requirement is to have a function that takes the
    predicted values and the ground truth values as input and returns a scalar.

    Args:
        loss (Callable): Loss function that is used for training.
    """

    @eqx.filter_value_and_grad
    def grad_loss(
        diff_model,
        static_model,
        ti: jax.Array,
        yi: jax.Array,
        y0i: jax.Array,
        penalties: Penalties,
        mask: jax.Array,
    ):
        """Calculates the masked loss of the model.

        Args:
            diff_model: Differentiable part of the model.
            static_model: Static part of the model.
            ti (jax.Array): Batch of times.
            yi (jax.Array): Batch of data.
            y0i (jax.Array): Batch of initial conditions.
            penalties (Penalties): Regularization penalties.
            mask (jax.Array): Float mask of shape ``(n_times,)`` weighting
                each time point's contribution to the loss. Use all-ones for
                no dropout (e.g., during validation).

        Returns:
            Scalar loss value.
        """

        model = eqx.combine(diff_model, static_model)
        y_pred = jax.vmap(model, in_axes=(0, 0))(ti, y0i)

        # Per-point loss; broadcast mask across batch and state dims.
        # Expected shape of `loss(y_pred, yi)` is (batch, time, n_states).
        per_point_loss = loss(y_pred, yi)
        broadcast_mask = mask[None, :, None]
        masked = per_point_loss * broadcast_mask

        # Normalize by kept points so the gradient scale is invariant in
        # `temporal_dropout_p`.
        denom = broadcast_mask.sum() * per_point_loss.shape[0] * per_point_loss.shape[2]
        loss_value = masked.sum() / denom

        return loss_value + penalties(model)

    @eqx.filter_jit
    def make_step(
        ti,
        yi,
        y0i,
        model,
        opt_state,
        optimizer,
        partitioned_model,
        penalties,
        mask,
    ):
        """Calculates the loss, gradient and updates the model.

        Args:
            ti (jax.Array): Batch of times.
            yi (jax.Array): Batch of data.
            y0i (jax.Array): Batch of initial conditions.
            model (NeuralODE): NeuralODE model to train.
            opt_state: State of the optimizer.
            optimizer: Optimizer of this session.
            partitioned_model: Tuple of (diff_model, static_model).
            penalties: Regularization penalties.
            mask (jax.Array): Temporal dropout mask of shape ``(n_times,)``.
        """

        diff_model, static_model = partitioned_model
        loss, grads = grad_loss(
            diff_model,
            static_model,
            ti,
            yi,
            y0i,
            penalties,
            mask,
        )

        # Pass params to optimizer.update() for optimizers that require it (e.g., weight decay)
        diff_params = eqx.filter(diff_model, eqx.is_inexact_array)
        updates, opt_state = optimizer.update(grads, opt_state, diff_params)
        # Apply updates to diff_model, then combine back with static_model
        diff_model = eqx.apply_updates(diff_model, updates)
        model = eqx.combine(diff_model, static_model)

        return loss, model, opt_state

    return make_step, grad_loss


def _log_progress(strat_index, step, loss, mae, log):
    if log is not None:
        with open(log, "a") as log_file:
            log_file.write(f"{strat_index + 1}\t{step}\t{loss}\t{mae}\n")


def _calculate_metrics(
    strat: Step,
    model: NeuralBase,
    times: jax.Array,
    data: jax.Array,
    inital_conditions: jax.Array,
    grad_loss,
) -> Tuple[jax.Array, jax.Array]:
    """Calculates the loss and MAE of a model on a dataset (no dropout).

    Args:
        strat (Step): The differentiation strategy to use.
        model (NeuralBase): The neural model to evaluate.
        times (jax.Array): The time points of the dataset.
        data (jax.Array): The data points of the dataset.
        inital_conditions (jax.Array): The initial conditions of the model.
        grad_loss: The gradient of the loss function to use.

    Returns:
        Tuple[jax.Array, jax.Array]: The loss and MAE of the model on the dataset.
    """
    diff_model, static_model = strat._partition_model(model)
    # Validation/metrics use an all-ones mask: every time point counted.
    full_mask = jnp.ones((times.shape[1],), dtype=jnp.float32)
    loss, _ = grad_loss(
        diff_model,
        static_model,
        times,
        data,
        inital_conditions,
        Penalties(),
        full_mask,
    )

    preds = jax.vmap(model, in_axes=(0, 0))(times, inital_conditions)  # type: ignore

    mae = jnp.mean(jnp.abs(data - preds))

    return loss, mae


def dataloader(arrays, batch_size, *, key):
    """Dataloader for training.

    Args:
        arrays (Tuple[Array]): Arrays to be batched.
        batch_size (int): Size of each batch.
        key (PRNG): Key for shuffling.

    Yields:
        Tuple[Array]: Batched arrays.
    """

    dataset_size = arrays[0].shape[0]

    assert all(array.shape[0] == dataset_size for array in arrays)
    indices = jnp.arange(dataset_size)
    while True:
        perm = jrandom.permutation(key, indices)
        (key,) = jrandom.split(key, 1)
        start = 0
        end = batch_size
        while end < dataset_size:
            batch_perm = perm[start:end]
            yield tuple(array[batch_perm] for array in arrays)
            start = end
            end = start + batch_size


def _serialize_milestone(
    model: NeuralBase,
    milestone_dir: str,
    strat_index: int,
    strategy: Dict,
):
    """Serializes a NeuralODE model.

    Args:
        model (NeuralODE): Model to serialize.

    Returns:
        Dict: Serialized model.
    """

    model.save_to_eqx(
        path=milestone_dir,
        name=f"run_{str(datetime.now())}_strategy_{strat_index + 1}",
        **{"strategy": strategy},
    )


def _scale_weights(model: T, scale: float) -> T:
    """Rescales weights and biases for models with small rates"""

    num_layers = len(model.func.mlp.layers)
    scaled_weights = [
        layer.weight * scale
        for layer in model.func.mlp.layers
        if hasattr(layer, "weight")
    ]
    scaled_biases = [
        layer.bias * scale  # type: ignore
        for layer in model.func.mlp.layers[:-1]
        if hasattr(layer, "bias")
    ]
    replacements = tuple(scaled_weights + scaled_biases)

    def loc_fun(tree):
        return tuple(
            [
                tree.func.mlp.layers[i].weight
                for i in range(num_layers)
                if hasattr(tree.func.mlp.layers[i], "weight")
            ]
            + [
                tree.func.mlp.layers[i].bias
                for i in range(num_layers - 1)
                if hasattr(tree.func.mlp.layers[i], "bias")
            ]
        )

    return eqx.tree_at(loc_fun, model, replace=replacements)
