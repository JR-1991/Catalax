import json
import equinox as eqx


def save(filename, hyperparams, model):
    with open(filename, "wb") as f:
        print(json.dumps(hyperparams, default=str))
        hyperparam_str = json.dumps(hyperparams, default=str)
        f.write((hyperparam_str + "\n").encode())
        eqx.tree_serialise_leaves(f, model)
