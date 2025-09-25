from __future__ import annotations

import hashlib
import json
import os
import zipfile
from datetime import datetime
from io import StringIO
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import mlcroissant as mlc
import pandas as pd
import rich

if TYPE_CHECKING:
    from catalax import Dataset, Measurement


def dataset_to_croissant(
    dataset: Dataset,
    dirpath: str,
    name: Optional[str] = None,
    license: str = "CC BY-SA 4.0",
    version: str = "1.0.0",
    cite_as: Optional[str] = None,
    url: Optional[str] = None,
    date_published: datetime = datetime.now(),
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
        name (str, optional): The name of the dataset. Defaults to None.
        date_published (datetime): The date the dataset was published.
        dataset (Dataset): The Dataset to export.
        dirpath (str): The directory to save the Croissant archive to.
        license (str, optional): The license for the dataset. Defaults to "CC BY-SA 4.0".
        version (str, optional): The version of the dataset. Defaults to "1.0.0".
        cite_as (Optional[str], optional): The citation for the dataset. Defaults to None.
        url (Optional[str], optional): The URL for the dataset. Defaults to None.
    """

    record_sets = []
    distribution = []
    zip_path = os.path.join(dirpath, f"{name}.zip")

    # Open ZIP file for writing
    with zipfile.ZipFile(zip_path, "w") as f:
        for measurement in dataset.measurements:
            file_obj, handler = _create_file_object(measurement)

            # Write the file to the ZIP
            f.writestr(file_obj.content_url, handler.getvalue())  # type: ignore

            meas_rec = _create_measurement_record_set(measurement, file_obj.id)
            inits_rec = _create_initial_condition_record_set(measurement)
            record_sets += [meas_rec, inits_rec]
            distribution.append(file_obj)

        metadata = mlc.Metadata(
            name=dataset.name or "Untitled Dataset",
            description=dataset.description,
            cite_as=cite_as,
            url=url,
            distribution=distribution,  # type: ignore
            record_sets=record_sets,
            license=license,  # type: ignore
            version=version,
            date_published=date_published,
        )

        # Write the metadata to the ZIP
        content = metadata.to_json()
        content = json.dumps(content, indent=2, default=str) + "\n"
        f.writestr("croissant.json", content)

    rich.print(f"🥐 Dataset exported to Croissant archive at {zip_path}")


def extract_record_set(
    croissant: mlc.Dataset,
    uuid_filter=lambda x: True,
) -> Dict[str, mlc.Records]:
    """Extracts the record sets from a Croissant dataset.

    Args:
        croissant (mlc.Dataset): The Croissant dataset.
        uuid_filter (Callable, optional): A filter function to apply to the UUIDs. Defaults to lambda x: True.

    Returns:
        Dict[str, mlc.RecordSet]: The record sets keyed by their UUID minus `/inits`.
    """

    return {
        rs.uuid.replace("/inits", ""): croissant.records(record_set=rs.uuid)
        for rs in croissant.metadata.record_sets
        if uuid_filter(rs.uuid)
    }


def json_lines_to_dict(record_set: mlc.Records) -> Dict[str, List[float]]:
    """Converts a given record set to a dictionary.

    Args:
        record_set (List[mlc.Records]): The record set to convert.

    Returns:
        Dict[str, List[float]]: The record set as a dictionary.
    """

    rows: List[Dict[str, float]] = list(record_set)  # type: ignore

    df = pd.DataFrame(rows)
    data = df.to_dict(orient="list")  # type: ignore

    return {key.split("/", 1)[-1]: value for key, value in data.items()}  # type: ignore


def _md5_hash_file(handler: StringIO) -> str:
    """Creates an MD5 hash of a file.

    In the case of the croissant export, hashes are computed in memory.

    Args:
        handler (StringIO): The file handler to hash.

    """

    hash = hashlib.md5(handler.getvalue().encode("utf-8")).hexdigest()

    handler.seek(0)
    return hash


def _create_file_object(measurement: Measurement) -> Tuple[mlc.FileObject, StringIO]:
    """Creates a FileObject from a Measurement.

    Args:
        measurement (Measurement): The measurement to create the FileObject from.

    Returns:
        mlc.FileObject: The FileObject.
    """

    # Set up the file objects
    fname = f"{measurement.id}.jsonl"
    handler = StringIO()
    handler.write(measurement.to_jsonl())
    handler.seek(0)

    # Get the hash
    hash = _md5_hash_file(handler)

    if measurement.description is None:
        measurement.description = (
            "JSON Lines file containing measurement data and initial conditions."
        )
    if measurement.name is None:
        measurement.name = measurement.id

    file_obj = mlc.FileObject(
        id=f"{measurement.id}_data",
        name=measurement.name,
        description=measurement.description,
        encoding_formats=["application/jsonlines"],
        content_url=fname,
        md5=hash,
    )

    return file_obj, handler


def _create_measurement_record_set(
    measurement: Measurement,
    file_object_id: str,
) -> mlc.RecordSet:
    """Creates a RecordSet from a Measurement.

    Args:
        measurement (Measurement): The measurement to create the RecordSet from.

    Returns:
        mlc.RecordSet: The RecordSet.
    """

    if measurement.description is None:
        measurement.description = f"Measurement Data for {measurement.id}"
    if measurement.name is None:
        measurement.name = measurement.id

    fields = [
        mlc.Field(
            id=f"{measurement.id}/{state}",
            name=state,
            description=f"The measurements of {state}.",
            data_types=mlc.DataType.FLOAT,  # type: ignore
            source=mlc.Source(
                file_object=file_object_id,
                extract=mlc.Extract(
                    column=state,
                ),
            ),
        )
        for state in measurement.data.keys()
    ]

    fields.append(
        mlc.Field(
            id=f"{measurement.id}/time",
            name="time",
            description="The time points of the measurements.",
            data_types=mlc.DataType.FLOAT,  # type: ignore
            source=mlc.Source(
                file_object=file_object_id,
                extract=mlc.Extract(
                    column="time",
                ),
            ),
        )
    )

    return mlc.RecordSet(
        id=measurement.id,
        name=measurement.name,
        description=measurement.description,
        fields=fields,
    )


def _create_initial_condition_record_set(measurement: Measurement) -> mlc.RecordSet:
    """Creates a RecordSet from a Measurement.

    Args:
        measurement (Measurement): The measurement to create the RecordSet from.

    Returns:
        mlc.RecordSet: The RecordSet.
    """

    if measurement.name is None:
        measurement.name = measurement.id

    # Create field objects for each state in initial conditions
    fields = [
        mlc.Field(
            id=f"{measurement.id}/inits/{state}",
            name=state,
            description=f"The initial conditions of {state}.",
            data_types=mlc.DataType.FLOAT,  # type: ignore
        )
        for state in measurement.initial_conditions.keys()
    ]

    # Prepare data with field IDs matching the expected structure
    data_with_prefixed_keys = {}
    for state, value in measurement.initial_conditions.items():
        data_with_prefixed_keys[f"{measurement.id}/inits/{state}"] = value

    return mlc.RecordSet(
        id=f"{measurement.id}/inits",
        name=f"{measurement.id}_inits",
        description=f"Initial Conditions for {measurement.id}",
        data=[data_with_prefixed_keys],
        fields=fields,
    )
