import pyenzyme as pe

import catalax as ctx

doc = pe.read_enzymeml("enzymeml.omex")

model, dataset = ctx.from_enzymeml(doc)
