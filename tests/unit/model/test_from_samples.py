import numpy as np

import catalax as ctx


def _toy_model() -> ctx.Model:
    model = ctx.Model(name="toy")
    model.add_state("A")
    model.add_ode("A", "-k_cat * A / (K_M + A)")
    model.parameters["k_cat"].value = 1.0
    model.parameters["k_cat"].initial_value = 1.0
    model.parameters["K_M"].value = 0.5
    model.parameters["K_M"].initial_value = 0.5
    return model


def test_from_samples_accepts_1d_dict_after_arviz_v1():
    # mcmc.get_samples() (without group_by_chain=True) returns flat 1-D
    # arrays per parameter. ArviZ v1's az.hdi rejects raw 1-D input —
    # Model.from_samples must reshape internally before calling az.hdi.
    rng = np.random.default_rng(0)
    samples = {
        "k_cat": rng.lognormal(0.0, 0.3, size=512),
        "K_M": rng.lognormal(-0.5, 0.3, size=512),
    }

    model = _toy_model()
    fitted = model.from_samples(samples, hdi_prob=0.95)

    for name in ("k_cat", "K_M"):
        param = fitted.parameters[name]
        assert param.hdi is not None
        assert param.hdi.q == 0.95
        assert param.hdi.lower <= param.value <= param.hdi.upper
        assert param.hdi.lower_50 <= param.value <= param.hdi.upper_50
