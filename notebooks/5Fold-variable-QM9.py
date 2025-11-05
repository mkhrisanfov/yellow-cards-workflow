# %%
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from catboost import CatBoostRegressor, Pool
from rdkit import Chem
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from torch_geometric.loader import DataLoader as GDataLoader
from tqdm.auto import tqdm

from yellow_cards_workflow import BASE_DIR
from yellow_cards_workflow.datasets import (
    FCD_Dataset,
    FCFP_Dataset,
    GNNIMDataset,
    get_dict_gnn_dataset,
    prepare_gnn_dataset,
    CNN_Dataset,
)
from yellow_cards_workflow.utils import (
    encode_smiles,
    generate_fingerprints,
    generate_rdkit_descriptors,
)
from yellow_cards_workflow.models import CNN, GNN, FCD, FCFP

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

SEED = 42


def train_fn(model, optim, loss_fn, epochs, train_dl, eval_dl, name):
    writer = SummaryWriter(log_dir=BASE_DIR / "logs" / name, flush_secs=15)

    torch.cuda.empty_cache()
    b_train = 1e5
    b_eval = 1e5

    bar = tqdm(range(epochs), leave=False, position=1)
    for epoch in bar:
        epoch_train_loss = model.train_fn(optim, loss_fn, train_dl)
        b_train = min(b_train, epoch_train_loss)

        epoch_eval_loss = model.eval_fn(loss_fn, eval_dl)
        b_eval = min(b_eval, epoch_eval_loss)

        bar.set_postfix_str(
            f"{epoch_train_loss:.3f}({b_train:.3f}) | {epoch_eval_loss:.3f}({b_eval:.3f})"
        )

        writer.add_scalar("loss/train", epoch_train_loss, epoch)
        writer.add_scalar("loss/eval", epoch_eval_loss, epoch)
        if epoch_eval_loss <= b_eval:
            torch.save(model.state_dict(), BASE_DIR / f"models/{name}.pth")
    bar.close()
    return b_train, b_eval


def prediction_loop(
    mu,
    mod_rate,
    morgan_fingerprints,
    rdkit_fingerprints,
    rdkit_descriptors,
    smiles_dict,
    encoded_smiles,
    gnn_num_fingerprints,
    gnn_fingerprints,
    gnn_mol_bonds,
):
    print(f"Started sequence for mu = {mu}, mod_rate = {mod_rate}.")
    properties_df = pd.read_csv(
        BASE_DIR / f"data/processed/qm9-variable-mu-{mu}-{mod_rate}.csv",
        sep=";",
        index_col=0,
    )
    molecular_properties = properties_df["g298_atom"].to_numpy() / (-1000)
    # %%
    kf = KFold(5, shuffle=True, random_state=SEED)
    predicted_values = {
        "smiles": [],
        "values": [],
        "cnn": [],
        "gnn": [],
        "fcfp": [],
        "fcd": [],
        "cb": [],
    }
    pbar = tqdm(kf.split(np.arange(len(smiles))), position=0, total=5)
    for k, (unique_train_idx, unique_eval_idx) in enumerate(pbar):
        train_gnn_fingerprints, eval_gnn_fingerprints = (
            [gnn_fingerprints[i] for i in unique_train_idx],
            [gnn_fingerprints[i] for i in unique_eval_idx],
        )
        train_gnn_mol_bonds, eval_gnn_mol_bonds = (
            [gnn_mol_bonds[i] for i in unique_train_idx],
            [gnn_mol_bonds[i] for i in unique_eval_idx],
        )
        train_encoded_smiles, eval_encoded_smiles = (
            encoded_smiles[unique_train_idx],
            encoded_smiles[unique_eval_idx],
        )
        train_rdkit_descriptors, eval_rdkit_descriptors = (
            rdkit_descriptors[unique_train_idx],
            rdkit_descriptors[unique_eval_idx],
        )
        train_morgan_fingerprints, eval_morgan_fingerprints = (
            morgan_fingerprints[unique_train_idx],
            morgan_fingerprints[unique_eval_idx],
        )
        train_rdkit_fingerprints, eval_rdkit_fingerprints = (
            rdkit_fingerprints[unique_train_idx],
            rdkit_fingerprints[unique_eval_idx],
        )
        train_molecular_properties, eval_molecular_properties = (
            molecular_properties[unique_train_idx],
            molecular_properties[unique_eval_idx],
        )

        predicted_values["smiles"].extend(smiles[unique_eval_idx])
        predicted_values["values"].extend(eval_molecular_properties)

        # pbar.write("Split Finished")
        model = CNN(
            len(smiles_dict) + 1,
            n_conv_layers=3,
            kernel_size=5,
            conv_channels=512,
            n_lin_layers=3,
        ).to(device)
        eval_ds = CNN_Dataset(eval_encoded_smiles, eval_molecular_properties)
        eval_dl = DataLoader(
            eval_ds,
            batch_size=512,
            shuffle=False,
            num_workers=4,
            persistent_workers=True,
        )
        train_ds = CNN_Dataset(train_encoded_smiles, train_molecular_properties)
        train_dl = DataLoader(
            train_ds,
            batch_size=512,
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
        )
        # pbar.write("CNN Loaded")
        b_train, b_eval = train_fn(
            model=model,
            optim=torch.optim.Adam(model.parameters(), lr=1e-4),
            loss_fn=nn.L1Loss(reduction="sum"),
            epochs=200,
            train_dl=train_dl,
            eval_dl=eval_dl,
            name=f"5fold-variable-QM9-{mod_rate}-{mu}-cnn-{SEED}-{k}",
        )
        pbar.write(f"5Fold\t CNN\t {k}\t {b_train:.4f}\t\t {b_eval:.4f}")
        model.load_state_dict(
            torch.load(
                BASE_DIR
                / f"models/5fold-variable-QM9-{mod_rate}-{mu}-cnn-{SEED}-{k}.pth",
                map_location=device,
                weights_only=True,
            )
        )
        predicted_values["cnn"].extend(
            model.eval_fn(nn.L1Loss(reduction="sum"), eval_dl, return_predictions=True)
        )

        train_gnn_dataset = GNNIMDataset(
            get_dict_gnn_dataset(
                train_gnn_fingerprints, train_gnn_mol_bonds, train_molecular_properties
            )
        )
        eval_gnn_dataset = GNNIMDataset(
            get_dict_gnn_dataset(
                eval_gnn_fingerprints, eval_gnn_mol_bonds, eval_molecular_properties
            )
        )
        model = GNN(
            gnn_num_fingerprints,
            embed_fingerprints=16,
            n_conv_layers=9,
            conv_channels=1024,
            n_lin_layers=10,
        ).to(device)
        eval_dl = GDataLoader(
            eval_gnn_dataset,
            batch_size=128,
            shuffle=False,
            num_workers=4,
            persistent_workers=True,
        )
        train_dl = GDataLoader(
            train_gnn_dataset,
            batch_size=128,
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
        )
        b_train, b_eval = train_fn(
            model=model,
            optim=torch.optim.Adam(model.parameters(), lr=1e-4),
            loss_fn=nn.L1Loss(reduction="sum"),
            epochs=200,
            train_dl=train_dl,
            eval_dl=eval_dl,
            name=f"5fold-variable-QM9-{mod_rate}-{mu}-gnn-{SEED}-{k}",
        )

        pbar.write(f"5Fold\t GNN\t {k}\t {b_train:.4f}\t\t {b_eval:.4f}")
        model.load_state_dict(
            torch.load(
                BASE_DIR
                / f"models/5fold-variable-QM9-{mod_rate}-{mu}-gnn-{SEED}-{k}.pth",
                map_location=device,
                weights_only=True,
            )
        )
        predicted_values["gnn"].extend(
            model.eval_fn(nn.L1Loss(reduction="sum"), eval_dl, return_predictions=True)
        )

        model = FCFP(n_layers=7, hidden_wts=4096).to(device)
        eval_ds = FCFP_Dataset(
            eval_morgan_fingerprints, eval_rdkit_fingerprints, eval_molecular_properties
        )
        eval_dl = DataLoader(
            eval_ds,
            batch_size=512,
            shuffle=False,
            num_workers=4,
            persistent_workers=True,
        )
        train_ds = FCFP_Dataset(
            train_morgan_fingerprints,
            train_rdkit_fingerprints,
            train_molecular_properties,
        )
        train_dl = DataLoader(
            train_ds,
            batch_size=512,
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
        )
        b_train, b_eval = train_fn(
            model=model,
            optim=torch.optim.Adam(model.parameters(), lr=1e-4),
            loss_fn=nn.L1Loss(reduction="sum"),
            epochs=250,
            train_dl=train_dl,
            eval_dl=eval_dl,
            name=f"5fold-variable-QM9-{mod_rate}-{mu}-fcfp-{SEED}-{k}",
        )
        pbar.write(f"5Fold\t FCFP\t {k}\t {b_train:.4f}\t\t {b_eval:.4f}")
        model.load_state_dict(
            torch.load(
                BASE_DIR
                / f"models/5fold-variable-QM9-{mod_rate}-{mu}-fcfp-{SEED}-{k}.pth",
                map_location=device,
                weights_only=True,
            )
        )
        predicted_values["fcfp"].extend(
            model.eval_fn(nn.L1Loss(reduction="sum"), eval_dl, return_predictions=True)
        )

        model = FCD(n_layers=7, hidden_wts=4096).to(device)
        eval_ds = FCD_Dataset(eval_rdkit_descriptors, eval_molecular_properties)
        eval_dl = DataLoader(
            eval_ds,
            batch_size=512,
            shuffle=False,
            num_workers=4,
            persistent_workers=True,
        )
        train_ds = FCD_Dataset(train_rdkit_descriptors, train_molecular_properties)
        train_dl = DataLoader(
            train_ds,
            batch_size=512,
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
        )
        b_train, b_eval = train_fn(
            model=model,
            optim=torch.optim.Adam(model.parameters(), lr=1e-4),
            loss_fn=nn.L1Loss(reduction="sum"),
            epochs=250,
            train_dl=train_dl,
            eval_dl=eval_dl,
            name=f"5fold-variable-QM9-{mod_rate}-{mu}-fcd-{SEED}-{k}",
        )
        pbar.write(f"5Fold\t FCD\t {k}\t {b_train:.4f}\t\t {b_eval:.4f}")
        model.load_state_dict(
            torch.load(
                BASE_DIR
                / f"models/5fold-variable-QM9-{mod_rate}-{mu}-fcd-{SEED}-{k}.pth",
                map_location=device,
                weights_only=True,
            )
        )
        predicted_values["fcd"].extend(
            model.eval_fn(nn.L1Loss(reduction="sum"), eval_dl, return_predictions=True)
        )

        eval_ds = Pool(
            np.hstack(
                [
                    eval_morgan_fingerprints,
                    eval_rdkit_fingerprints,
                    eval_rdkit_descriptors,
                ]
            ),
            eval_molecular_properties,
        )
        trn_ds = Pool(
            np.hstack(
                [
                    train_morgan_fingerprints,
                    train_rdkit_fingerprints,
                    train_rdkit_descriptors,
                ]
            ),
            train_molecular_properties,
        )
        model = CatBoostRegressor(
            loss_function="MAE",
            task_type="GPU",
            devices="0",
            metric_period=20,
            learning_rate=0.00116,
            depth=8,
            iterations=10000,
            use_best_model=True,
            silent=True,
            allow_writing_files=False,
        )
        model.fit(trn_ds, eval_set=eval_ds, plot=False)
        model.save_model(
            BASE_DIR / f"models/5fold-variable-QM9-{mod_rate}-{mu}-cb-{SEED}-{k}.cb"
        )
        b_train = model.get_best_score().get("learn", {"MAE": np.nan})["MAE"]
        b_eval = model.get_best_score().get("validation", {"MAE": np.nan})["MAE"]
        pbar.write(f"5Fold\t CB\t {k}\t {b_train:.4f}\t\t {b_eval:.4f}")
        model = CatBoostRegressor().load_model(
            BASE_DIR / f"models/5fold-variable-QM9-{mod_rate}-{mu}-cb-{SEED}-{k}.cb"
        )
        predicted_values["cb"].extend(np.stack(model.predict(eval_ds)))

    predicted_df = pd.DataFrame(predicted_values)
    predicted_df.to_csv(
        BASE_DIR
        / f"data/processed/qm9-variable-mu-predictions-{mu}-{mod_rate}-{SEED}.csv",
        sep=";",
    )
    print(f"Finished sequence for mu = {mu}, mod_rate = {mod_rate}.")


if __name__ == "__main__":
    print("Start Preparation")
    supplier = Chem.SDMolSupplier(BASE_DIR / "data/input/gdb9.sdf")
    print("Loading molecules")
    molecules = np.array([x for x in tqdm(supplier)])
    print(np.count_nonzero(molecules))
    clean_mask = np.load(BASE_DIR / "data/processed/qm9-mask.npy")
    molecules = molecules[clean_mask]
    noniso_smiles = np.array(
        list(
            map(
                lambda x: Chem.MolToSmiles(x, isomericSmiles=False, canonical=True),
                tqdm(molecules),
            )
        )
    )
    smiles = np.array(
        list(
            map(
                lambda x: Chem.MolToSmiles(x, isomericSmiles=True, canonical=True),
                tqdm(molecules),
            )
        )
    )
    print(len(np.unique(noniso_smiles)), len(np.unique(smiles)))
    print("Generating fingerprints")
    morgan_fingerprints, rdkit_fingerprints = generate_fingerprints(molecules)
    print("Generating descriptors")
    rdkit_descriptors = generate_rdkit_descriptors(molecules)
    smiles_dict, encoded_smiles = encode_smiles(smiles)
    print("Generating GNN dataset")
    gnn_num_fingerprints, gnn_fingerprints, gnn_mol_bonds = prepare_gnn_dataset(
        molecules
    )

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(device)

    for mod_rate in [0.15, 0.25]:
        for mu in [2.5]:
            prediction_loop(
                mu,
                mod_rate,
                morgan_fingerprints,
                rdkit_fingerprints,
                rdkit_descriptors,
                smiles_dict,
                encoded_smiles,
                gnn_num_fingerprints,
                gnn_fingerprints,
                gnn_mol_bonds,
            )
