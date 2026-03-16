import os

import numpy as np
import jax.numpy as jnp
import h5py


def load_dict_from_hdf5(file_name):
    """
    Load the contents of an HDF5 file (written by save_dict_to_hdf5) back
    into a dictionary.

    Scalars are recovered from dataset attributes, and arrays are loaded as
    NumPy arrays.

    Parameters
    ----------
    file_name : str
        Path to the HDF5 file.

    Returns
    -------
    dict
        Dictionary with the same keys and values that were originally saved.

    Raises
    ------
    FileNotFoundError
        If *file_name* does not exist.
    """
    if not os.path.exists(file_name):
        raise FileNotFoundError(f"File not found: {file_name}")

    data_dict = {}
    with h5py.File(file_name, "r") as hdf:
        for key in hdf.keys():
            dset = hdf[key]
            if "value" in dset.attrs:
                # Scalar stored as an attribute on an empty dataset
                data_dict[key] = dset.attrs["value"]
            else:
                arr = np.array(dset)
                # Unwrap 0-d arrays to Python scalars
                data_dict[key] = arr.item() if arr.ndim == 0 else arr
    return data_dict


def save_dict_to_hdf5(file_name, data_dict):
    """
    Save the contents of a dictionary to an HDF5 file, handling different data types.

    Parameters:
    - file_name: The name of the HDF5 file to be created.
    - data_dict: The dictionary to save, where keys are dataset names and values are the data.
    """
    with h5py.File(file_name, "w") as hdf:
        for key, value in data_dict.items():
            # Check the type of the value to handle it appropriately
            if isinstance(value, np.ndarray):
                # Save NumPy arrays directly as datasets
                hdf.create_dataset(key, data=value)
            elif isinstance(value, jnp.ndarray):
                # Convert JAX array to NumPy array and save it
                hdf.create_dataset(key, data=np.array(value))
            elif isinstance(
                value, (float, int, np.float32, np.float64, np.int32, np.int64)
            ):
                # Save floats or integers as attributes of a dataset
                # If the value is a NumPy scalar, convert to a native Python type
                if hasattr(value, "item"):
                    value = value.item()
                dset = hdf.create_dataset(key, data=[])
                dset.attrs["value"] = value
            else:
                raise TypeError(f"Unsupported data type for key '{key}': {type(value)}")
