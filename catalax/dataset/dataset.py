import warnings
import pandas as pd
import jax
import jax.numpy as jnp

from pydantic import BaseModel, Field
from typing import List, Optional, Tuple

from measurement import Measurement

class Dataset(BaseModel):

    """A class to represent a dataset.

    This class hosts multiple measurements of an experiment and can be
    used with other Catalax functionalities.

    """

    species: List[str] = Field(
        ...,
        description="List of species within this dataset.",
    )

    name: Optional[str] = Field(
        default=None,
        description="Name or ID of the dataset.",
    )

    description: Optional[str] = Field(
        default=None,
        description="Additional textual information of the dataset."
    )

    measurements: List[Measurement] = Field(
        default_factory=list,
        description="A list of all measurements found in this dataset"
    )

    # ! Adders

    def add_measurement(self, measurement: Measurement):
        """Adds a measurement to the dataset.

        Args:
            measurement (ctx.Measurement): The measurement object to add.

        """

        assert not any(meas.id == measurement.id for meas in self.measurements), (
            f"Measurement with ID={measurement.id} already exists."
        )

        unused_species = [sp for sp in measurement.data.keys() if sp not in self.species]
        missing_species = [sp for sp in self.species if sp not in measurement.data.keys()]

        if unused_species:
            warnings.warn(f"Species {unused_species} are not used in the dataset.")

        if missing_species:
            raise ValueError(
                f"The measurement species are inconsistent with the dataset species. Missing {missing_species}"
            )

        self.measurements.append(measurement)

    # ! Exporters

    def to_dataframe(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Exports a dataset into two DataFrames (measurement)"""

        data = pd.concat([meas.to_dataframe() for meas in self.measurements])
        inits = pd.DataFrame(
            {
                "measurementId": meas.id,
                **meas.initial_concentrations
            }
            for meas in self.measurements
        )

        return data, inits

    def to_jax_array(self, species_order: List[str]):
        """Converts the dataset to a JAX array"""

        data = jnp.stack(
            [
                meas.to_jax_array(species_order=species_order)
                for meas in self.measurements
            ],
            axis=0,
        )

        time = jnp.stack(
            [meas.time for meas in self.measurements], # type: ignore
            axis=0,
        )
        initial_concentrations = [meas.initial_concentrations for meas in self.measurements]

        return data, time, initial_concentrations

    # ! Importers

    @classmethod
    def from_dataframe(
        cls,
        data: pd.DataFrame,
        inits: pd.DataFrame,
    ):
        """Creates a dataset from Pandas DataFrames

        - Data is expected in the following format:
            measurementId | time | species1 | species2 ...

        - Inits are expected in the following format:
            measurementId | species1 | species2

        Please note, that 'measurementId' should be consistent and
        non-ambigous in bot tables. Consistency will be checked though.
        Also, the inits table should have unique 'measurementId' entries.
        Duplicates will throw an exception.

        Example:

            data = [
                {"A": 0.0, "time": 0.0, "measurementId": "m1"},
                {"A": 0.5, "time": 1.0, "measurementId": "m1"},
                {"A": 1.0, "time": 2.0, "measurementId": "m1"},
                {"A": 1.0, "time": 0.0, "measurementId": "m2"},
                {"A": 2.5, "time": 1.0, "measurementId": "m2"},
                {"A": 4.5, "time": 2.0, "measurementId": "m2"},
            ]

            inits = [
                {"A": 0.0, "measurementId": "m1"},
                {"A": 1.0, "measurementId": "m2"},
            ]

            data = pd.DataFrame(data)
            inits = pd.DataFrame(inits)

            Dataset.from_dataframe(data=data, inits=inits)

        Args:
            data (pandas.DataFrame): Contains all time courses.
            inits (pandas.DataFrame): Containse all initial concentrations.

        """

        assert "measurementId" in data.columns, "Missing column in data table: 'measurementId'"
        assert "time" in data.columns, "Missing column in data table: 'time'"
        assert "measurementId" in inits.columns, "Missing column in inits table: 'measurementId'"

        # Check if IDs are consistents
        data_ids = set(data["measurementId"])
        init_ids = set(inits["measurementId"])

        id_diff = set(list(data_ids.difference(init_ids)) + list(init_ids.difference(data_ids)))

        if id_diff:
            missing_in_data = [id_ for id_ in id_diff if id_ not in data["measurementId"].to_list()]
            missing_in_inits = [id_ for id_ in id_diff if id_ not in inits["measurementId"].to_list()]

            raise ValueError(
                f"Measurement IDs are incosistent in between both tables:\n\t- Data: {missing_in_data}\n\t- Inits: {missing_in_inits}"
            )

        # Initialize dataset
        species = [sp for sp in inits.columns if sp != "measurementId"]
        dataset = cls(species=species)

        # Extract data and inits by measurement IDs
        for meas_id in data_ids:
            sub_inits = inits[inits.measurementId == meas_id].to_dict("records")[0] # type: ignore
            sub_data = data[data.measurementId == meas_id]
            meas_id = sub_inits.pop("measurementId")

            dataset.add_measurement(
                measurement=Measurement.from_dataframe(
                    df=sub_data, # type: ignore
                    initial_concentrations=sub_inits,
                    id=meas_id,
                )
            )

        return dataset

if __name__ == "__main__":
    dataset = Dataset(
        name="My measurement",
        species=["A", "B", "C"]
    )
