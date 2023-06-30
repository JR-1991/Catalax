from datetime import datetime
import os
from typing import Dict, Optional, Tuple

import tqdm
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
import optax

from .utils import save


def train_neural_ode(
    model: "NeuralODE",
    data: jax.Array,
    times: jax.Array,
    inital_conditions: jax.Array,
    batch_size: int,
    steps_strategy: Tuple[int, ...],
    lr_strategy: Tuple[float, ...],
    length_strategy: Tuple[float, ...],
    optimizer=optax.adabelief,
    print_every: int = 100,
    log: Optional[str] = None,
    save_milestones: bool = False,
    milestone_dir: str = "./milestones",
):
    # Set up PRNG keys
    key = jrandom.PRNGKey(420)
    _, _, loader_key = jrandom.split(key, 3)
    _, length_size, _ = data.shape

    # Set up logger
    if log is not None:
        log_file = open(log, "w")
        log_file.write("strategy\tstep\tmean_loss\nmae\n")
        log_file.close()

    # Set up milestone directory
    if save_milestones:
        os.makedirs(milestone_dir, exist_ok=True)

    for strat_index, (lr, steps, length) in enumerate(
        zip(lr_strategy, steps_strategy, length_strategy)
    ):
        print(
            f"<< Strategy #{strat_index+1}: Learning rate: {lr} | Steps: {steps} Length: {length*100}% >>\n"
        )

        # Prepare optimizer per strategy
        optimizer = optax.adabelief(lr)
        opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

        # Apply training strategy
        max_time = int(length_size * length) + 1

        if max_time == 1:
            # Make sure that at least two steps are taken
            max_time = 2

        _times = times[:, :max_time]
        _data = data[:, :max_time, :]

        # Prepare data generator
        batches = dataloader(
            (_data, inital_conditions, _times), batch_size, key=loader_key
        )

        # Set up progress bar
        pbar = tqdm.tqdm(total=steps, desc=f"Startup")

        for step, (yi, y0i, ti) in zip(range(steps), batches):
            loss, model, opt_state = make_step(
                ti=ti,
                yi=yi,
                y0i=y0i,
                model=model,
                opt_state=opt_state,
                optimizer=optimizer,
            )

            if (step % print_every) == 0 or step == steps - 1:
                # Calculate mean loss over data
                loss, _ = grad_loss(model, _times, _data, inital_conditions)
                preds = jax.vmap(model, in_axes=(0, 0))(_times, inital_conditions)
                mae = jnp.mean(jnp.abs(_data - preds))

                pbar.update(print_every)
                pbar.set_description(f"MSE: {loss:.4f} MAE: {mae:.4f}")

                if log is not None:
                    with open(log, "a") as log_file:
                        log_file.write(f"{strat_index+1}\t{step}\t{loss}\n{mae}\n")

        pbar.close()
        print("\n")

        if save_milestones:
            # Save milestone
            _serialize_milestone(
                model,
                milestone_dir,
                strat_index,
                {"lr": lr, "steps": steps, "length": length},
            )

    return model


def _serialize_milestone(
    model: "NeuralODE",
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

    path = os.path.join(
        milestone_dir, f"run_{str(datetime.now())}_strategy_{strat_index+1}.eqx"
    )
    save(path, {"hyperparameters": model.hyperparams, "strategy": strategy}, model)


@eqx.filter_value_and_grad
def grad_loss(model: "NeuralODE", ti: jax.Array, yi: jax.Array, y0i: jax.Array):
    """Calculates the L2 loss of the model.

    Args:
        model (NeuralODE): NeuralODE model to train.
        ti (jax.Array): Batch of times.
        yi (jax.Array): Batch of data.
        y0i (jax.Array): Batch of initial conditions.

    Returns:
        float: Average L2 loss.
    """
    y_pred = jax.vmap(model, in_axes=(0, 0))(ti, y0i)
    return jnp.mean((yi - y_pred) ** 2)


@eqx.filter_jit
def make_step(ti, yi, y0i, model, opt_state, optimizer):
    """Calculates the loss, gradient and updates the model.

    Args:
        ti (jax.Array): Batch of times.
        yi (jax.Array): Batch of data.
        y0i (jax.Array): Batch of initial conditions.
        model (NeuralODE): NeuralODE model to train.
        opt_state (...): State of the optimizer.
        optimizer (...): Optimizer of this session.
    """

    loss, grads = grad_loss(model, ti, yi, y0i)
    updates, opt_state = optimizer.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return loss, model, opt_state


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
