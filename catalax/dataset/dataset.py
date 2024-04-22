import os
import uuid
import zipfile
import rich
import warnings
import pandas as pd
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import mlcroissant as mlc

from datetime import datetime
from copy import deepcopy
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Tuple, Union

from .croissant import extract_record_set, json_lines_to_dict, dataset_to_croissant
from .measurement import Measurement


class Dataset(BaseModel):
    """A class to represent a dataset.

    This class hosts multiple measurements of an experiment and can be
    used with other Catalax functionalities.

    """

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier of the dataset.",
    )

    species: List[str] = Field(
        ...,
        description="List of species within this dataset.",
    )

    name: Optional[str] = Field(
        default=None,
        description="Name or ID of the dataset.",
    )

    description: Optional[str] = Field(
        default=None, description="Additional textual information of the dataset."
    )

    measurements: List[Measurement] = Field(
        default_factory=list,
        description="A list of all measurements found in this dataset",
    )

    # ! Adders
    def add_measurement(self, measurement: Measurement) -> None:
        """Adds a measurement to the dataset.

        Args:
            measurement (ctx.Measurement): The measurement object to add.

        """

        assert not any(
            meas.id == measurement.id for meas in self.measurements
        ), f"Measurement with ID={measurement.id} already exists."

        unused_species = [
            sp for sp in measurement.data.keys() if sp not in self.species
        ]
        missing_species = [
            sp for sp in self.species if sp not in measurement.data.keys()
        ]

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
            {"measurementId": meas.id, **meas.initial_conditions}
            for meas in self.measurements
        )

        return data, inits

    def to_jax_arrays(
        self,
        species_order: List[str],
        inits_to_array: bool = False,
    ) -> Tuple[jax.Array, jax.Array, Union[jax.Array, Dict[str, float]]]:
        """Converts the dataset to a JAX arrays

        This method requires a species order to arrange the data in the correct order.

        - The shape of the data is (n_measurements, n_time_points, n_species).
        - The shape of the time is (n_measurements, n_time_points).
        - The initial conditions is (n_measurements, n_species).

        Args:
            species_order (List[str]): The order of the species in the array.

        Returns:
            Tuple[jax.Array, jax.Array, List[Dict[str, float]]]: The data array, time array, and initial conditions.
        """

        data = []
        time = []
        initial_conditions = []

        for meas in self.measurements:
            data_, time_, inits = meas.to_jax_arrays(
                species_order=species_order,
                inits_to_array=inits_to_array,
            )

            data.append(data_)
            time.append(time_)
            initial_conditions.append(inits)

        if inits_to_array:
            return (
                jnp.stack(data, axis=0),
                jnp.stack(time, axis=0),
                jnp.stack(initial_conditions, axis=0),
            )
        else:
            return (
                jnp.stack(data, axis=0),
                jnp.stack(time, axis=0),
                initial_conditions,  # type: ignore
            )

    def to_croissant(
        self,
        dirpath: str,
        license: str = "CC BY-SA 4.0",
        version: str = "1.0.0",
        cite_as: Optional[str] = None,
        url: Optional[str] = None,
        date_published: datetime = datetime.now(),  # type: ignore
    ) -> None:
        """Exports a Dataset to a Croissant (https://github.com/mlcommons/croissant) archive.

        The croissant format is a JSON-LD standard to describe datasets and enable
        interoperability between different tools and platforms. The format is
        designed to be simple and easy to use, while still providing the necessary
        information to describe a dataset.

        This implementation breaks up all measurements found in a dataset into
        individual JSON Lines files, and then zips them up with a metadata file in
        the Croissant format. Initial conditions are stored within the croissant metadata.

        Catalax support a method to read in Croissant archives, so this method can be
        used to export a dataset to a format that can be read back in. However, you can also
        use the official Croissant tools to read in the dataset. For this to do, you
        can use the `mlcommons/croissant` Python package to parse the unzipped files.

        Args:
            dataset (Dataset): The Dataset to export.
            dirpath (str): The directory to save the Croissant archive to.
            license (str, optional): The license for the dataset. Defaults to "CC BY-SA 4.0".
            version (str, optional): The version of the dataset. Defaults to "1.0.0".
            cite_as (Optional[str], optional): The citation for the dataset. Defaults to None.
            url (Optional[str], optional): The URL for the dataset. Defaults to None.
            date_published (datetime, optional): The date the dataset was published. Defaults to datetime.now().
        """

        from .croissant import dataset_to_croissant

        os.makedirs(dirpath, exist_ok=True)

        dataset_to_croissant(
            dataset=self,
            dirpath=dirpath,
            license=license,
            version=version,
            cite_as=cite_as,
            url=url,
            date_published=date_published,  # type: ignore
        )

        archive_path = os.path.join(dirpath, self.name.replace("_", "")) + ".zip"

        rich.print(f"🥐 Dataset exported to Croissant archive at {archive_path}")

    # ! Importers

    @classmethod
    def from_dataframe(
        cls,
        name: str,
        data: pd.DataFrame,
        inits: pd.DataFrame,
        id: Optional[str] = None,
        description: Optional[str] = "",
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

        assert (
            "measurementId" in data.columns
        ), "Missing column in data table: 'measurementId'"
        assert "time" in data.columns, "Missing column in data table: 'time'"
        assert (
            "measurementId" in inits.columns
        ), "Missing column in inits table: 'measurementId'"

        if id is None:
            id = str(uuid.uuid4())

        # Check if IDs are consistents
        data_ids = set(data["measurementId"])
        init_ids = set(inits["measurementId"])

        id_diff = set(
            list(data_ids.difference(init_ids)) + list(init_ids.difference(data_ids))
        )

        if id_diff:
            missing_in_data = [
                id_ for id_ in id_diff if id_ not in data["measurementId"].to_list()
            ]
            missing_in_inits = [
                id_ for id_ in id_diff if id_ not in inits["measurementId"].to_list()
            ]

            raise ValueError(
                f"Measurement IDs are incosistent in between both tables:\n\t- Data: {missing_in_data}\n\t- Inits: {missing_in_inits}"
            )

        # Initialize dataset
        species = [sp for sp in inits.columns if sp != "measurementId"]
        dataset = cls(
            species=species,
            name=name,
            description=description,
            id=id,
        )

        # Extract data and inits by measurement IDs
        for meas_id in data_ids:
            sub_inits = inits[inits.measurementId == meas_id].to_dict("records")[0]  # type: ignore
            sub_data = data[data.measurementId == meas_id]
            meas_id = sub_inits.pop("measurementId")

            dataset.add_measurement(
                measurement=Measurement.from_dataframe(
                    df=sub_data,  # type: ignore
                    initial_conditions=sub_inits,
                    id=meas_id,
                )
            )

        return dataset

    @classmethod
    def from_croissant(cls, path: str):
        """Reads a Croissant archive and returns a Dataset.

        Args:
            path (str): The path to the Croissant archive.

        Returns:
            Dataset: The Dataset object.
        """

        with zipfile.ZipFile(path, "r") as f:
            f.extractall(".")

        croissant_path = os.path.join(".", "croissant.json")
        croissant_ds = mlc.Dataset(jsonld=croissant_path)

        init_rs = extract_record_set(croissant_ds, lambda uuid: "/inits" in uuid)
        meas_rs = extract_record_set(croissant_ds, lambda uuid: "/inits" not in uuid)

        species = set()
        measurements = []

        for uuid, rs in meas_rs.items():
            assert uuid in init_rs, f"Initial conditions not found for {uuid}"

            # Extract the initial conditions and measurements
            inits = list(init_rs[uuid])[0]
            meas_recs = croissant_ds.records(record_set=uuid)

            data = json_lines_to_dict(meas_recs)
            time = data.pop("time")

            # Update the species set
            species.update(inits.keys())

            # Create a Measurement object
            measurements.append(
                Measurement(
                    initial_conditions=inits,
                    data=data,  # type: ignore
                    time=time,
                    id=uuid,
                )
            )

        return cls(
            id=croissant_ds.metadata.id,
            name=croissant_ds.metadata.name,
            description=croissant_ds.metadata.description,
            species=list(species),
            measurements=measurements,
        )  # type: ignore

    # ! Plotting
    def plot(
        self,
        ncols: int = 2,
        show: bool = True,
        path: Optional[str] = None,
        model: Optional["Model"] = None,  # type: ignore
        measurement_ids: List[str] = [],
    ):
        """Plots all measurements in the dataset.

        Args:
            ncols (int, optional): Number of columns in the plot. Defaults to 2.
            show (bool, optional): Whether to show the plot. Defaults to True.

        Returns:
            matplotlib.figure.Figure: The figure object
        """

        if len(measurement_ids) == 0:
            measurement_ids = [meas.id for meas in self.measurements]

        if len(measurement_ids) == 1:
            nrows = 1
            ncols = 1
        else:
            nrows = int(np.ceil(len(measurement_ids) / ncols))

        fig, axs = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(ncols * 5, nrows * 3),
        )

        if len(measurement_ids) > 1:
            axs = axs.flatten()
        else:
            axs = [axs]

        for i, meas in enumerate(self.measurements):
            if meas.id not in measurement_ids:
                continue

            meas.plot(ax=axs[i])

        # Remove legend from plots that have a right neighbor
        for i, ax in enumerate(axs):
            if i % ncols == ncols - 1:
                ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
            else:
                ax.legend().set_visible(False)

        plt.tight_layout(w_pad=4, h_pad=4)

        if path:
            plt.savefig(
                path,
                dpi=300,
                format="png",
                bbox_inches="tight",
            )
        elif show:
            plt.show()

        return fig

    # ! Data Augmentation
    def augment(
        self,
        n_augmentations: int,
        sigma: float = 0.5,
        seed: int = 42,
    ):
        """Augments the dataset by adding Gaussian noise to the data.

        Args:
            n_augmentations (int): The number of augmentations to be performed.
            sigma (float, optional): The standard deviation of the Gaussian noise. Defaults to 0.5.
            seed (int, optional): The seed for the random number generator. Defaults to 42.

        Returns:
            Dataset: The augmented dataset.
        """

        augmented_dataset = deepcopy(self)  # type: ignore
        augmented_meas = []

        for i in range(n_augmentations):
            for meas in augmented_dataset.measurements:
                augmented_meas.append(
                    meas.augment(
                        sigma=sigma,
                        seed=seed + i,
                    )
                )

        augmented_dataset.measurements += augmented_meas

        return augmented_dataset

    # ! Utilities
    @staticmethod
    def _get_vmap_dims(
        data: jax.Array,
        time: jax.Array,
        y0s: Union[jax.Array, Dict[str, float]],
    ) -> Tuple[Optional[int], None, Optional[int]]:
        """Determines the axes in which vmap will be applies"""

        if isinstance(y0s, dict):
            multiple_y0s = len(y0s) > 1
        elif isinstance(y0s, jax.Array):
            multiple_y0s = len(y0s.shape) == 2
        else:
            raise TypeError(f"Expected dict or array for 'y0s' but got {type(y0s)}")

        return (
            0 if len(data.shape) == 3 else None,
            None,
            0 if multiple_y0s else None,
        )
