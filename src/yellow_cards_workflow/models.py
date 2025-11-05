import torch
import torch.nn as nn
import torch_geometric.nn as gnn


class CNN(nn.Module):
    """
    A Convolutional Neural Network (CNN) model designed for processing sequences of symbols.

    Parameters:
    n_sym (int): The number of unique symbols in the input data.
    n_conv_layers (int, optional): The number of convolutional layers.
    kernel_size (int, optional): The size of the convolutional kernel.
    conv_channels (int, optional): The number of channels in each convolutional layer.
    n_lin_layers (int, optional): The number of linear layers.
    """

    def __init__(self, n_sym,
                 n_conv_layers: int = 2,
                 kernel_size: int = 6,
                 conv_channels: int = 300,
                 n_lin_layers: int = 2):
        super().__init__()
        self.n_conv_layers = n_conv_layers
        self.kernel_size = kernel_size
        self.conv_channels = conv_channels
        self.n_lin_layers = n_lin_layers
        self.embed_atoms = nn.Embedding(
            n_sym, n_sym, padding_idx=0).from_pretrained(torch.eye(n_sym), freeze=False)

        self.conv_in = nn.Sequential(
            nn.Conv1d(n_sym, conv_channels, kernel_size, padding="same"),
            nn.ReLU(inplace=True),
        )

        n_conv_layers = max(1, n_conv_layers)
        self.conv_layers = nn.ModuleList(
            [self._make_conv_layer(conv_channels, conv_channels, kernel_size)
                for _ in range(n_conv_layers-1)]
        )

        self.lin0 = nn.Sequential(
            nn.Linear(conv_channels, 2*conv_channels),
            nn.ReLU(inplace=True),
        )

        n_lin_layers = max(2, n_lin_layers)
        self.lin_layers = nn.ModuleList(
            [self._make_lin_layer(2*conv_channels, 2*conv_channels)
             for _ in range(n_lin_layers-2)]
        )
        self.out = nn.Sequential(
            nn.Linear(2*conv_channels, 1),
            nn.Identity(),
        )

    def _make_conv_layer(self, in_channnels, out_channels, kernel_size):
        return nn.Sequential(
            nn.Conv1d(in_channnels, out_channels, kernel_size, padding="same"),
            nn.ReLU(inplace=True)
        )

    def _make_lin_layer(self, in_features, out_features):
        return nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        Forward pass of the model.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor with shape (batch_size, num_atoms).

        Returns:
        --------
        torch.Tensor
            Output tensor after passing through the model layers.
        """
        x = torch.permute(self.embed_atoms(x), (0, 2, 1))
        x = self.conv_in(x)
        for i, conv in enumerate(self.conv_layers):
            x = conv(x)
        x = torch.sum(x, dim=2).squeeze()
        x = self.lin0(x)
        for i, lin in enumerate(self.lin_layers):
            x = lin(x)
        return self.out(x).squeeze()

    def train_fn(self, optim, loss_fn, train_dl):
        """
        Train the model for one epoch.

        Parameters:
        -----------
        optim : torch.optim.Optimizer
            The optimizer to use for training.
        loss_fn : callable
            The loss function to compute the training loss.
        train_dl : DataLoader
            The data loader providing batches of training data.

        Returns:
        --------
        float
            The average training loss per sample over the epoch.
        """
        self.train()
        epoch_train_loss = 0
        for encoded_smiles, molecular_properties in train_dl:
            optim.zero_grad()
            pred_molecular_properties = self.forward(
                encoded_smiles.to(
                    next(self.parameters()).device, non_blocking=True))
            loss = loss_fn(
                pred_molecular_properties,
                molecular_properties.to(next(self.parameters()).device, non_blocking=True))
            loss.backward()
            optim.step()
            epoch_train_loss += loss.detach().cpu()
        return epoch_train_loss/len(train_dl.dataset)

    def eval_fn(self, loss_fn, eval_dl, return_predictions=False):
        """
        Evaluate the model on a validation or test dataset.

        Parameters:
        -----------
        loss_fn : callable
            The loss function to compute the evaluation loss.
        eval_dl : DataLoader
            The data loader providing batches of evaluation data.
        return_predictions : bool, optional
            If True, returns the predicted retention indices. Defaults to False.

        Returns:
        --------
        float or numpy.ndarray
            If `return_predictions` is False, returns the average evaluation loss over the dataset.
            If `return_predictions` is True, returns a NumPy array of predicted retention indices.
        """
        self.eval()
        epoch_eval_loss = 0
        all_pred_molecular_properties = []
        with torch.no_grad():
            for encoded_smiles, molecular_properties in eval_dl:
                pred_molecular_properties = self.forward(
                    encoded_smiles.to(next(self.parameters()).device, non_blocking=True))
                if return_predictions:
                    all_pred_molecular_properties.append(
                        pred_molecular_properties)
                loss = loss_fn(
                    pred_molecular_properties,
                    molecular_properties.to(next(self.parameters()).device, non_blocking=True))
                epoch_eval_loss += loss.detach().cpu()
        if return_predictions:
            return torch.cat(all_pred_molecular_properties, 0).cpu().numpy()
        else:
            return epoch_eval_loss/len(eval_dl.dataset)


class FCD(nn.Module):
    """
    A fully connected deep neural network model.

    Parameters:
    ----------
    n_layers : int, optional (default=5)
        Number of hidden layers in the network.

    hidden_wts : int, optional (default=2048)
        Number of neurons in each hidden layer.

    device : torch.device, optional (default=torch.device("cuda"))
        Device to run the model on (e.g., CPU or GPU).
    """

    def __init__(self, n_layers=5, hidden_wts=2048, device=torch.device("cuda")):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_wts = hidden_wts
        self.device = device
        self.fc_in = nn.Sequential(
            nn.Linear(217, hidden_wts),
            nn.ReLU(inplace=True),
        )
        self.hidden_linear = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(hidden_wts, hidden_wts),
                              nn.ReLU(inplace=True)) for _ in range(n_layers)
            ]
        )

        self.fc_out = nn.Sequential(
            nn.Linear(hidden_wts, 1),
            nn.Identity(),
        )

    def forward(self, rdkit_descriptors):
        """
        Forward pass of the model.

        Parameters:
        ----------
        rdkit_descriptors : torch.Tensor
            Input tensor of RDKit Descriptors with shape (batch_size, 217).

        Returns:
        -------
        torch.Tensor
            Output tensor with shape (batch_size, 1).
        """
        x = self.fc_in(rdkit_descriptors)
        for i, lin in enumerate(self.hidden_linear):
            x = lin(x)
        return self.fc_out(x)

    def train_fn(self, optim, loss_fn, train_dl):
        """
        Train the model for one epoch.

        Parameters:
        -----------
        optim : torch.optim.Optimizer
            The optimizer to use for training.
        loss_fn : callable
            The loss function to compute the loss between predictions and true values.
        train_dl : DataLoader
            The data loader providing batches of training data.

        Returns:
        --------
        float
            The average training loss over the epoch.
        """
        self.train()
        epoch_train_loss = 0
        for rdkit_descriptors, molecular_properties in train_dl:
            optim.zero_grad()
            pred_molecular_properties = self.forward(
                rdkit_descriptors.to(self.device,
                                     non_blocking=True))
            loss = loss_fn(
                pred_molecular_properties,
                molecular_properties.to(self.device, non_blocking=True))
            loss.backward()
            optim.step()
            epoch_train_loss += loss.detach().cpu()
        return epoch_train_loss/len(train_dl.dataset)

    def eval_fn(self, loss_fn, eval_dl, return_predictions=False):
        """
        Evaluate the model on a dataset.

        Parameters:
        -----------
        loss_fn : callable
            The loss function to compute the loss between predictions and true values.
        eval_dl : DataLoader
            The data loader providing batches of evaluation data.
        return_predictions : bool, optional (default=False)
            Whether to return the predicted values.

        Returns:
        --------
        float or numpy.ndarray
            If `return_predictions` is False, returns the average evaluation loss over the dataset.
            If `return_predictions` is True, returns a NumPy array of predicted values.
        """
        self.eval()
        epoch_eval_loss = 0
        all_pred_molecular_properties = []
        with torch.no_grad():
            for descriptors, molecular_properties in eval_dl:
                pred_molecular_properties = self.forward(
                    descriptors.to(
                        self.device, non_blocking=True))
                if return_predictions:
                    all_pred_molecular_properties.append(pred_molecular_properties)
                loss = loss_fn(
                    pred_molecular_properties,
                    molecular_properties.to(self.device, non_blocking=True))
                epoch_eval_loss += loss.detach().cpu()
        if return_predictions:
            return torch.cat(all_pred_molecular_properties, 0).cpu().numpy()
        else:
            return epoch_eval_loss/len(eval_dl.dataset)


class FCFP(nn.Module):
    """
    A Fully Connected Feedforward Perceptron (FCFP) model for chemical compound property prediction.

    Parameters:
        n_layers (int): Number of hidden layers in the network.
        hidden_wts (int): Number of neurons in each hidden layer.
        device (torch.device): Device to run the model on, e.g., "cuda" or "cpu".
    """

    def __init__(self, n_layers=5, hidden_wts=4096, device=torch.device("cuda")):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_wts = hidden_wts
        self.device = device
        self.fc_in = nn.Sequential(
            nn.Linear(2048, hidden_wts),
            nn.ReLU(inplace=True),
        )
        self.hidden_linear = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(hidden_wts, hidden_wts),
                              nn.ReLU(inplace=True)) for _ in range(n_layers)
            ]
        )

        self.fc_out = nn.Sequential(
            nn.Linear(hidden_wts, 1),
            nn.Identity(),
        )

    def forward(self, morgan_fingerprints, rdkit_fingerprints):
        """
        Forward pass of the FCFP model.

        Parameters:
            morgan_fingerprints (torch.Tensor): Morgan fingerprints of shape (batch_size, 1024).
            rdkit_fingerprints (torch.Tensor): RDKit fingerprints of shape (batch_size, 1024).

        Returns:
            torch.Tensor: Predicted values of shape (batch_size, 1).
        """
        x = torch.cat([morgan_fingerprints, rdkit_fingerprints], dim=1)
        x = self.fc_in(x)
        for i, lin in enumerate(self.hidden_linear):
            x = lin(x)
        return self.fc_out(x)

    def train_fn(self, optim, loss_fn, train_dl):
        """
        Trains the FCFP model on a given training dataset.

        Parameters:
            optim (torch.optim.Optimizer): Optimizer for updating the model weights.
            loss_fn (callable): Loss function to compute the difference
                                between predictions and true values.
            train_dl (torch.utils.data.DataLoader): DataLoader containing the training data.

        Returns:
            float: Average training loss over the entire dataset.
        """
        self.train()
        epoch_train_loss = 0
        for morgan_fingerprints, rdkit_fingerprints, molecular_properties in train_dl:
            optim.zero_grad()
            pred_molecular_properties = self.forward(
                morgan_fingerprints.to(self.device,
                                       non_blocking=True),
                rdkit_fingerprints.to(self.device,
                                      non_blocking=True))
            loss = loss_fn(
                pred_molecular_properties,
                molecular_properties.to(self.device, non_blocking=True))
            loss.backward()
            optim.step()
            epoch_train_loss += loss.detach().cpu()
        return epoch_train_loss/len(train_dl.dataset)

    def eval_fn(self, loss_fn, eval_dl, return_predictions=False):
        """
        Evaluates the FCFP model on a given evaluation dataset.

        Parameters:
            loss_fn (callable): Loss function to compute the difference
                                between predictions and true values.
            eval_dl (torch.utils.data.DataLoader): DataLoader containing the evaluation data.
            return_predictions (bool): Whether to return the predicted values
                                        or just the average loss.

        Returns:
            np.ndarray or float: Predicted values as a NumPy array
                                if `return_predictions` is True,
                                otherwise, the average evaluation loss over the entire dataset.
        """
        self.eval()
        epoch_eval_loss = 0
        all_pred_molecular_properties = []
        with torch.no_grad():
            for morgan_fingerprints, rdkit_fingerprints, molecular_properties in eval_dl:
                pred_molecular_properties = self.forward(
                    morgan_fingerprints.to(self.device,
                                           non_blocking=True),
                    rdkit_fingerprints.to(self.device,
                                          non_blocking=True))
                if return_predictions:
                    all_pred_molecular_properties.append(pred_molecular_properties)
                loss = loss_fn(
                    pred_molecular_properties,
                    molecular_properties.to(self.device, non_blocking=True))
                epoch_eval_loss += loss.detach().cpu()
        if return_predictions:
            return torch.cat(all_pred_molecular_properties, 0).cpu().numpy()
        else:
            return epoch_eval_loss/len(eval_dl.dataset)


class GNN(nn.Module):
    """
    A Graph Neural Network (GNN) model designed for molecular fingerprint classification.

    Parameters:
    n_fingerprints (int): Number of unique fingerprints.
    embed_fingerprints (int, optional): Dimensionality of the fingerprint embedding.
    n_conv_layers (int, optional): Number of convolutional layers.
    conv_channels (int, optional): Number of channels in each convolutional layer.
    n_lin_layers (int, optional): Number of linear layers after the convolutional layers.
    """

    def __init__(self, n_fingerprints,
                 embed_fingerprints: int = 64,
                 n_conv_layers: int = 5,
                 conv_channels: int = 256,
                 n_lin_layers: int = 2):
        super().__init__()
        self.n_fingerprints = n_fingerprints
        self.embed_fingerprints = embed_fingerprints
        self.n_conv_layers = n_conv_layers
        self.conv_channels = conv_channels
        self.n_lin_layers = n_lin_layers

        self.embed = nn.Embedding(n_fingerprints, embed_fingerprints)

        self.in_conv = self._make_layer(embed_fingerprints, conv_channels)
        self.conv_layers = nn.ModuleList(
            [self._make_layer(conv_channels, conv_channels) for _ in range(n_conv_layers)])

        self.lin0 = nn.Sequential(
            nn.Linear(conv_channels, 2*conv_channels),
            nn.ReLU(),
        )

        self.lin_layers = nn.ModuleList(
            [self._make_lin_layer(2*conv_channels, 2*conv_channels) for _ in range(n_lin_layers)])
        self.out = nn.Sequential(
            nn.Linear(2*conv_channels, 1),
            nn.Identity()
        )

    def _make_layer(self, in_channels, out_channels):
        return gnn.Sequential(
            'x, edge_index', [
                (gnn.GraphConv(in_channels, out_channels), 'x, edge_index -> x'),
                (nn.ReLU(inplace=True), 'x -> x'),
            ]
        )

    def _make_lin_layer(self, in_weights, hidden):
        return nn.Sequential(
            nn.Linear(in_weights, hidden),
            nn.ReLU(inplace=True),
        )

    def forward(self, data):
        """
        Forward pass of the GNN model.

        Parameters:
        data (torch_geometric.data.Data): Input graph data with attributes 'x' and 'edge_index'.

        Returns:
        torch.Tensor: Predicted retention indices.
        """
        x, edge_index = data.x, data.edge_index
        x = self.embed(x)

        x = self.in_conv(x, edge_index)
        for i, conv in enumerate(self.conv_layers):
            x = conv(x, edge_index)
        x = gnn.pool.global_add_pool(x, data.batch)
        x = self.lin0(x)
        for i, lin in enumerate(self.lin_layers):
            x = lin(x)

        return self.out(x)

    def train_fn(self, optim, loss_fn, train_dl):
        """
        Trains the model on a training dataset.

        Parameters:
        optim (torch.optim.Optimizer): Optimizer for updating model parameters.
        loss_fn (callable): Loss function to compute the loss.
        train_dl (DataLoader): DataLoader for the training dataset.

        Returns:
        float: Average training loss over the epoch.
        """
        self.train()
        epoch_train_loss = 0
        for data in train_dl:
            optim.zero_grad()
            pred_molecular_properties = self.forward(
                data.to(next(self.parameters()).device, non_blocking=True))
            loss = loss_fn(
                pred_molecular_properties.squeeze(),
                data.y)
            loss.backward()
            optim.step()
            epoch_train_loss += loss.detach().cpu()
        return epoch_train_loss/len(train_dl.dataset)

    def eval_fn(self, loss_fn, eval_dl, return_predictions=False):
        """
        Evaluates the model on an evaluation dataset.

        Parameters:
        loss_fn (callable): Loss function to compute the loss.
        eval_dl (DataLoader): DataLoader for the evaluation dataset.
        return_predictions (bool, optional): Whether to return predictions. Defaults to False.

        Returns:
        float or numpy.ndarray: Average evaluation loss over the epoch
                                or concatenated predictions if `return_predictions` is True.
        """
        self.eval()
        epoch_eval_loss = 0
        all_pred_molecular_properties = []
        with torch.no_grad():
            for data in eval_dl:
                pred_molecular_properties = self.forward(
                    data.to(next(self.parameters()).device, non_blocking=True))
                if return_predictions:
                    all_pred_molecular_properties.append(
                        pred_molecular_properties)
                loss = loss_fn(
                    pred_molecular_properties.squeeze(),
                    data.y)
                epoch_eval_loss += loss.detach().cpu()
        if return_predictions:
            return torch.cat(all_pred_molecular_properties, 0).cpu().numpy()
        else:
            return epoch_eval_loss/len(eval_dl.dataset)
