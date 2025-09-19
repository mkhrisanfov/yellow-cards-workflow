from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm.autonotebook import tqdm
from torch_geometric.data import Data, InMemoryDataset


class CNN_Dataset(Dataset):
    """
    A PyTorch Dataset class for handling encoded SMILES
    and their corresponding molecular properties.

    Parameters:
        encoded_smis (list of lists): Encoded SMILES representations.
        molecular_properties (list of float): molecular properties
                                              corresponding to the encoded SMILES.
    """

    def __init__(self, encoded_smis, molecular_properties):
        self.encoded_smis = torch.LongTensor(encoded_smis)
        self.molecular_properties = torch.FloatTensor(molecular_properties)

    def __getitem__(self, index):
        return (self.encoded_smis[index],  self.molecular_properties[index])

    def __len__(self):
        return len(self.molecular_properties)


class FCD_Dataset(Dataset):
    """
    A PyTorch Dataset class for handling molecular descriptors and mixed descriptors.

    Parameters:
        descriptors (list of lists): Molecular descriptors.
        mix_desc (list of lists): Mixed descriptors.
    """

    def __init__(self, descriptors, mix_desc):
        mix_desc = np.vstack(mix_desc)
        descriptors = np.vstack(descriptors)
        self.mix_desc = torch.FloatTensor(mix_desc)
        self.descriptors = torch.FloatTensor(descriptors)

    def __getitem__(self, index):
        return (self.descriptors[index], self.mix_desc[index])

    def __len__(self):
        return len(self.mix_desc)


class FCFP_Dataset(Dataset):
    """
    A PyTorch Dataset class for handling RDKit and Morgan fingerprints
    along with molecular properties.

    Parameters:
        rdkit_fingerprints (list of lists): RDKit fingerprints.
        morgan_fingerprints (list of lists): Morgan fingerprints.
        molecular_properties (list of float): molecular properties corresponding
                                              to the fingerprints.
    """

    def __init__(self, rdkit_fingerprints, morgan_fingerprints, molecular_properties):
        molecular_properties = np.vstack(molecular_properties)
        self.molecular_properties = torch.FloatTensor(molecular_properties)
        self.rdkit_fingerprints = torch.FloatTensor(
            np.vstack(rdkit_fingerprints))
        self.morgan_fingerprints = torch.FloatTensor(
            np.vstack(morgan_fingerprints))

    def __getitem__(self, index):
        return (self.morgan_fingerprints[index],
                self.rdkit_fingerprints[index],
                self.molecular_properties[index])

    def __len__(self):
        return len(self.molecular_properties)


class GNNIMDataset(InMemoryDataset):
    """
    A PyTorch Geometric InMemoryDataset class for handling graph data.

    Parameters:
        raw_data (list of dict): Raw data containing molecular graphs and their properties.

    Attributes:
        data (Data): PyG Data object containing the processed dataset.
        slices (dict): Slices information for the processed dataset.
    """

    def __init__(self, raw_data):
        super().__init__(None, None, None)
        self.data, self.slices = self.process_data(raw_data)

    @property
    def raw_file_names(self):
        return ['raw_data.pt']  # Expecting raw data file

    @property
    def processed_file_names(self):
        return ['data.pt']  # Processed dataset file

    def process_data(self, raw_data):
        # Load raw data (list of dicts)
        data_list = []  # List of `Data` objects

        for sample in raw_data:
            # Convert features to tensors
            # Fingerprints as node features
            x = torch.tensor(sample['fingerprints'], dtype=torch.long)
            edge_index = torch.tensor(
                sample['mol_bonds'], dtype=torch.long)  # Connectivity
            y = torch.tensor([sample['retention_time']],
                             dtype=torch.float)  # Retention time

            # Create PyG `Data` object
            data = Data(x=x, edge_index=edge_index, y=y)

            data_list.append(data)

        # Convert list of Data objects into a single `Data` object
        return self.collate(data_list)


def get_dict_gnn_dataset(fingerprints, mol_bonds, molecular_properties):
    """
    Converts fingerprints, molecular bonds, and molecular properties into a list of dictionaries.

    Parameters:
        fingerprints (list of lists): RDKit fingerprints.
        mol_bonds (list of lists): Molecular bond indices.
        molecular_properties (list of float): molecular properties corresponding to the molecules.

    Returns:
        list of dict: List of dictionaries containing molecular data.
    """
    dataset = []
    for i in tqdm(range(len(molecular_properties)), leave=False):
        new_mol_dict = {}
        new_mol_dict["fingerprints"] = fingerprints[i]
        new_mol_dict["mol_bonds"] = mol_bonds[i]
        new_mol_dict["retention_time"] = molecular_properties[i]
        dataset.append(new_mol_dict)
    return dataset


def prepare_gnn_dataset(molecules):
    """
    Prepares graph data for GNN training by extracting atom and bond features,
    and creating fingerprints.

    Parameters:
        molecules (list of RDKit Mol): List of RDKit molecule objects.

    Returns:
        tuple: A tuple containing the number of unique fingerprints, a list of fingerprints,
               and a list of molecular bonds.
    """
    atom_dict = defaultdict(lambda: len(atom_dict))
    bond_dict = defaultdict(lambda: len(bond_dict))
    edge_dict = defaultdict(lambda: len(edge_dict))
    hybdn_dict = defaultdict(lambda: len(hybdn_dict))
    fingerprint_dict = defaultdict(lambda: len(fingerprint_dict))

    all_fingerprints = []
    all_mol_bonds = []
    for m, mol in enumerate(tqdm(molecules, leave=False)):
        fingerprints = np.zeros(mol.GetNumAtoms(), dtype=np.int64)
        for a, atom in enumerate(mol.GetAtoms()):
            atom_str = "-".join(map(str, [atom.GetSmarts(),
                                          atom.GetTotalDegree(),
                                          atom.GetTotalNumHs(),
                                          atom.GetFormalCharge()]))
            neighbors_str = "".join(
                sorted([ngh.GetSmarts() for ngh in atom.GetNeighbors()]))
            fingerprints[a] = fingerprint_dict[f"{atom_str}_{neighbors_str}"]
        mol_bonds = np.zeros((2, 2*mol.GetNumBonds()), dtype=np.int64)
        for b, bond in enumerate(mol.GetBonds()):
            mol_bonds[0, b], mol_bonds[1,
                                       b] = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            mol_bonds[0, -b-1], mol_bonds[1, -b -
                                          1] = bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()

        all_fingerprints.append(fingerprints)
        all_mol_bonds.append(mol_bonds)
    print("Atom Dict", len(atom_dict))
    print("Hybdn Dict", len(hybdn_dict))
    print("FPs Dict", len(fingerprint_dict))
    num_fingerprints = len(fingerprint_dict)+1
    return num_fingerprints, all_fingerprints, all_mol_bonds
