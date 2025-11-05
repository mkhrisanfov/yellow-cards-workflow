import os
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Descriptors import CalcMolDescriptors
from sklearn.preprocessing import StandardScaler
from tqdm.autonotebook import tqdm
from tqdm.contrib.concurrent import thread_map
from yellow_cards_workflow import BASE_DIR


def load_predicted_df(
    filename, model_names, offset: float = 0, **kwargs
) -> pd.DataFrame:
    """
    Load a dataframe containing predicted values from a file.

    Args:
        filename (str): The name of the file to load the dataframe from.
        model_names (List[str]): A list of model names to include in the dataframe.
        offset (float): The offset to use when calculating the mean threshold error.

    Returns:
        pd.DataFrame: A dataframe containing the predicted values.
    """
    dataframe = pd.read_csv(filename, sep=";", index_col=0)
    for model in model_names:
        if dataframe[model].dtype in [str, object]:
            dataframe[model] = [
                float(str(x).strip("[]")) for x in dataframe[model].to_numpy()
            ]
        dataframe[model] += offset
    dataframe["values"] += offset
    dataframe = dataframe.sort_values("smiles").reset_index(drop=True)
    return dataframe


def encode_smiles(smiles):
    """
    Encode a list of SMILES strings into numerical format.

    Parameters
    ----------
    smiles : list of str
        A list of SMILES strings to encode.

    Returns
    -------
    encoded_smiles : torch.Tensor
        A tensor of encoded SMILES strings.
    """
    sym_dict = defaultdict(lambda: len(sym_dict) + 1)

    encoded_smiles = torch.zeros((len(smiles), 256), dtype=torch.long)
    for s, smile in enumerate(tqdm(smiles)):
        if "i" in smile or "l" in smile or "r" in smile:
            smile = smile.replace("Si", "X").replace("Cl", "Y").replace("Br", "Z")
        for s2, sym in enumerate(smile):
            encoded_smiles[s, s2] = sym_dict[sym]
    return sym_dict, encoded_smiles


def generate_fingerprints(molecules):
    """
    Generate fingerprints for a list of molecules.

    Parameters
    ----------
    molecules : list of rdkit.rdChem.Mol
        A list of molecules to generate fingerprints for.

    Returns
    -------
    morgan_fingerprints : numpy.ndarray
        An array of Morgan fingerprints for the input molecules.
    rdkit_fingerprints : numpy.ndarray
        An array of RDKit fingerprints for the input molecules.
    """
    # FCFP and CatBoost
    print("Generating Morgan fingerprints")
    morgan_generator = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=1024)
    morgan_fingerprints = np.array(
        thread_map(
            morgan_generator.GetCountFingerprintAsNumPy, molecules, chunksize=500
        )
    )

    # FCFP and CatBoost
    print("Generating RDKit fingerprints")
    rdkit_generator = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=1024)
    rdkit_fingerprints = np.array(
        thread_map(rdkit_generator.GetCountFingerprintAsNumPy, molecules, chunksize=500)
    )
    return morgan_fingerprints, rdkit_fingerprints


def generate_rdkit_descriptors(molecules, load=False):
    """
    Generate descriptors for a list of molecules.

    Parameters
    ----------
    molecules : list of rdkit.rdChem.Mol
        A list of molecules to generate descriptors for.

    Returns
    -------
    descriptors : numpy.ndarray
        An array of RDKit descriptors for the input molecules.
    """
    print("Loading / Generating RDKit descriptors")
    if not load or not os.path.exists(
        BASE_DIR / "data/processed/rdkit_descriptors.npy"
    ):
        descriptors = thread_map(CalcMolDescriptors, molecules, chunksize=500)
        descriptors = pd.DataFrame(descriptors).fillna(0)
        print(descriptors.shape)
        np.save(
            BASE_DIR / "data/processed/rdkit_descriptors.npy",
            descriptors.to_numpy(dtype=float, na_value=0),
        )
    else:
        descriptors = np.load(BASE_DIR / "data/processed/rdkit_descriptors.npy")

    print("Scaling RDKit descriptors to zero mean and unit variance")
    scl = StandardScaler()
    descriptors = scl.fit_transform(descriptors)
    return descriptors
