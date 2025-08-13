import pyenzyme as pe

doc = pe.EnzymeMLDocument(
    name="test",
)

# Small molecules
s1 = doc.add_to_small_molecules(
    id="s1",
    name="s1",
)
s2 = doc.add_to_small_molecules(
    id="s2",
    name="s2",
)

# Proteins
p1 = doc.add_to_proteins(
    id="p1",
    name="p1",
)

# Measurements
meas1 = doc.add_to_measurements(
    id="meas1",
    name="meas1",
)
meas1.add_to_species_data(
    species_id=s1.id,
    name="s1",
    initial=1,
    data=[1, 2, 3],
    time=[0, 1, 2],
)
meas1.add_to_species_data(
    species_id=s2.id,
    name="s2",
    initial=4,
    data=[4, 5, 6],
    time=[0, 1, 2],
)
meas1.add_to_species_data(
    species_id=p1.id,
    name="p1",
    initial=7,
    data=[7, 8, 9],
    time=[0, 1, 2],
)

meas2 = doc.add_to_measurements(
    id="meas2",
    name="meas2",
)
meas2.add_to_species_data(
    species_id=s1.id,
    name="s1",
    initial=1,
    data=[1, 2, 3],
    time=[0, 1, 2],
)
meas2.add_to_species_data(
    species_id=s2.id,
    name="s2",
    initial=4,
    data=[4, 5, 6],
    time=[0, 1, 2],
)
meas2.add_to_species_data(
    species_id=p1.id,
    name="p1",
    initial=7,
    data=[7, 8, 9],
    time=[0, 1, 2],
)

pe.write_enzymeml(doc, "tests/enzymeml_measurements_only.json")
