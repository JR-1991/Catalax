import rich
import catalax as ctx

dataset = ctx.Dataset(species=["A", "B", "C"])

measurement = ctx.Measurement(
    initial_conditions={
        "A": 1.0,
        "B": 4.0,
        "C": 7.0,
    },
    data={
        "A": [1, 2, 3],
        "B": [4, 5, 6],
        "C": [7, 8, 9],
    },
    time=[0, 1, 2],
)

dataset.add_measurement(measurement)

measurement = ctx.Measurement(
    initial_conditions={
        "A": 1.0,
        "B": 4.0,
        "C": 7.0,
    },
    data={
        "A": [1, 2, 3],
        "B": [4, 5, 6],
        "C": [7, 8, 9],
    },
    time=[0, 1, 2],
)

dataset.add_measurement(measurement)

df_meas, df_init = dataset.to_dataframe()

df_meas.to_csv("df_meas.csv", index=False)
df_init.to_csv("df_init.csv", index=False)

data, time, inits = dataset.to_jax_arrays(
    species_order=["A", "B", "C"]
)

rich.print(f"[bold]Data shape: {data.shape}[/bold]")
rich.print(f"[bold]Time shape: {time.shape}[/bold]")
rich.print(f"[bold]Inits: {inits}[/bold]")

f = measurement.plot(show=False)

f.savefig("measurement.png")
