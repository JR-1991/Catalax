import numpy as np
import pyenzyme as pe
from rich import print

import catalax as ctx

################
# 1. create EnzymeML Document
################

COMPOSE = False

if COMPOSE:
    # create vessel
    vessel = pe.Vessel(
        id="vessel",
        name="vessel",
        volume=1,
        unit="mL",  # type: ignore
    )

    # create EnzymeML document
    doc = pe.compose(
        name="Lactate dehydrogenase",
        reactions=["RHEA:19909"],
        proteins=["P00175"],
        vessel=vessel,
    )

    # export EnzymeML with reaction only, no kinetic law
    pe.write_enzymeml(doc, "examples/datasets/enzymeml_reaction_after_compose.json")

else:
    doc = pe.read_enzymeml("examples/datasets/enzymeml_reaction_after_compose.json")

################
# 2. add kinetic law via doc.reactions and doc.equations
################

# constant
# 2.1. add kinetic law via doc.reactions

# extract reaction species
substrate_reaction = doc.filter_reactions(id="RHEA:19909")[0]

protein = doc.filter_proteins(id="l_lactate_dehydrogenase_cytochrome")[0]
substrate = doc.filter_small_molecules(id="s_lactate")[0]
product = doc.filter_small_molecules(id="pyruvate")[0]

# add modifier to reaction
substrate_reaction.add_to_modifiers(
    species_id=protein.id,
    role=pe.ModifierRole.BIOCATALYST,
)

# add kinetic law to reaction
equation = "s_lactate * k_cat * l_lactate_dehydrogenase_cytochrome / (K_M + s_lactate)"

substrate_reaction.kinetic_law = pe.Equation(
    species_id=substrate.id,
    equation=equation,
    equation_type=pe.EquationType.ODE,
)

# duplicate reaction and change educt and product
product_reaction = doc.add_to_reactions(
    id="r2",
    name="Lactate dehydrogenase",
    kinetic_law=pe.Equation(
        species_id=product.id,
        equation=equation,
        equation_type=pe.EquationType.ODE,
    ),
    reactants=substrate_reaction.products,
    products=substrate_reaction.reactants,
    modifiers=substrate_reaction.modifiers,
)

# 2.2. add ODEs via doc.equations

# substrate reaction
doc.add_to_equations(
    species_id=substrate.id,
    equation="-s_lactate * k_cat * l_lactate_dehydrogenase_cytochrome / (K_M + s_lactate)",
    equation_type=pe.EquationType.ODE,
)

doc.add_to_equations(
    species_id=protein.id,
    equation="-k_inact * l_lactate_dehydrogenase_cytochrome",
    equation_type=pe.EquationType.ODE,
)

# product reaction
doc.add_to_equations(
    species_id=product.id,
    equation="s_lactate * k_cat * l_lactate_dehydrogenase_cytochrome / (K_M + s_lactate)",
    equation_type=pe.EquationType.ODE,
)

################
# 3. create Catalax model from EnzymeML and add data
################

# create model from EnzymeML
model = ctx.model.Model.from_enzymeml(doc)

# Set parameters
model.parameters["k_cat"].value = 1
model.parameters["K_M"].value = 180
model.parameters["k_inact"].value = 0.0042

# Create dataset and add initial conditions
dataset = ctx.Dataset.from_model(model)
dataset.add_initial(s_lactate=25, l_lactate_dehydrogenase_cytochrome=5, pyruvate=0)
dataset.add_initial(s_lactate=50, l_lactate_dehydrogenase_cytochrome=5, pyruvate=0)
dataset.add_initial(s_lactate=400, l_lactate_dehydrogenase_cytochrome=5, pyruvate=0)


# Configure and run simulation
config = ctx.SimulationConfig(t0=0, t1=180, nsteps=19)
results = model.simulate(dataset=dataset, config=config)

# add data to enzymeml
data_unit = "mmol/L"
time_unit = "s"

for meas_id, meas in enumerate(results.measurements):
    enzml_meas = doc.add_to_measurements(
        id=f"m{meas_id}",
        name=f"Measurement {meas_id}",
    )

    for sid, value in meas.data.items():
        noise = np.random.normal(0, 0.05, len(value))
        value = value + noise
        enzml_meas.add_to_species_data(
            species_id=sid,
            initial=value[0],  # type: ignore
            time=meas.time.tolist(),  # type: ignore
            data=value.tolist(),
            data_unit=data_unit,  # type: ignore
            time_unit=time_unit,  # type: ignore
            data_type=pe.DataTypes.CONCENTRATION,
        )

    # add data for constant species not in dataset
    for small_molecule in doc.small_molecules:
        if small_molecule.id not in meas.data:
            print(f"adding data for constant species {small_molecule.id}")
            enzml_meas.add_to_species_data(
                species_id=small_molecule.id,
                initial=5,
                data_unit=data_unit,  # type: ignore
                data_type=pe.DataTypes.CONCENTRATION,
            )

# write EnzymeML with equations to file
pe.write_enzymeml(doc, "examples/datasets/enzymeml_complete.json")
