import logging
import anndata as ad
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np
from scipy.sparse import csc_matrix
import scanpy as sc
import pandas as pd
from sklearn.preprocessing import binarize
import visualkeras 

# Set logging level
logging.basicConfig(level=logging.INFO)

## VIASH START
par = {
    'input_train_mod1': "/gpfs/scratch1/nodespecific/int5/rverplanke/data/phase2-private-data/predict_modality/openproblems_bmmc_cite_phase2_mod2/openproblems_bmmc_cite_phase2_mod2.censor_dataset.output_train_mod1.h5ad",
    
    'input_train_mod2': "/gpfs/scratch1/nodespecific/int5/rverplanke/data/phase2-private-data/predict_modality/openproblems_bmmc_cite_phase2_mod2/openproblems_bmmc_cite_phase2_mod2.censor_dataset.output_train_mod2.h5ad",
    
    'input_test_mod1': "/gpfs/scratch1/nodespecific/int5/rverplanke/data/phase2-private-data/predict_modality/openproblems_bmmc_cite_phase2_mod2/openproblems_bmmc_cite_phase2_mod2.censor_dataset.output_test_mod1.h5ad",
    
    'output': 'batchencoder_m_mod2.h5ad',   
}
## VIASH END

logging.info('Reading `h5ad` files...')
input_train_mod1 = ad.read_h5ad(par['input_train_mod1'])
input_train_mod2 = ad.read_h5ad(par['input_train_mod2'])
input_test_mod1 = ad.read_h5ad(par['input_test_mod1'])

mod1 = input_train_mod1.var['feature_types'][0]
mod2 = input_train_mod2.var['feature_types'][0]

# Combine train and test data for mod1
input_train = ad.concat(
    {"train": input_train_mod1, "test": input_test_mod1},
    axis=0,
    join="outer",
    label="group",
    fill_value=0,
    index_unique="-"
)


# Convert to numpy arrays
X_train = input_train_mod1.X.toarray()
y_train = input_train_mod2.X.toarray()
X_test = input_test_mod1.X.toarray()

# Multi Layer Perceptron model
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, min_val, max_val):
        super(MLP, self).__init__()
        self.min_val = min_val
        self.max_val = max_val
        self.layers = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(input_dim, 320),
            nn.ReLU(),
            nn.Linear(320, 460),
            nn.ReLU(),
            nn.Linear(460, 620),
            nn.ReLU(),
            nn.Linear(620, 440),
            nn.ReLU(),
            nn.Linear(440, output_dim)
        )
    
    def forward(self, x):
        x = self.layers(x)
        x = torch.clamp(x, self.min_val, self.max_val)
        return x
        
# Encoder-Decoder-LSTM model 
class EncoderDecoderLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, min_val, max_val):
        super(EncoderDecoderLSTM, self).__init__()
        self.min_val = min_val
        self.max_val = max_val
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Encode
        x = x.unsqueeze(1) 
        _, (hidden, cell) = self.encoder(x)
        
        # Decode
        decoder_input = torch.zeros((x.size(0), 1, hidden.size(2))).to(x.device)
        decoder_output, _ = self.decoder(decoder_input, (hidden, cell))
        
        # Output
        output = self.fc(decoder_output.squeeze(1))
        
        # Clamp the output
        output = torch.clamp(output, self.min_val, self.max_val)
        
        return output

        
# Normalize per batch
def normalize_per_batch(data, batches):
    normalized_data = []
    batch_means = []
    batch_sds = []
    
    for batch in batches:
        batch_data = data[data.obs['batch'] == batch]
        batch_array = batch_data.X.toarray()
        means = np.mean(batch_array)
        sds = np.std(batch_array)
        normalized_batch_data = (batch_array - means) / sds
        normalized_data.append(normalized_batch_data)
        batch_means.append(means)
        batch_sds.append(sds)
    
    return np.vstack(normalized_data), np.mean(batch_means), np.mean(batch_sds)

# Normalize training data
train_batches = input_train_mod1.obs.batch.unique()
X_train, train_means, train_sds = normalize_per_batch(input_train_mod1, train_batches)

# Normalize test data using training means and sds
def normalize_test_data(data, means, sds):
    data = data.copy()
    data_array = data.X.toarray()
    data = (data_array - means) / sds
    return data

input_test_mod1_norm = normalize_test_data(input_test_mod1, train_means, train_sds)
X_test = input_test_mod1_norm

# get min and max values for modality 2
min_val = np.min(input_train_mod2.X)
max_val = np.max(input_train_mod2.X)

input_dim = X_train.shape[1]
output_dim = y_train.shape[1]
hidden_dim = 256

# Choose model
#model = MLP(input_dim, output_dim, min_val, max_val)
#model = TransformerModel(input_dim, output_dim, min_val, max_val)
model = EncoderDecoderLSTM(input_dim, hidden_dim, output_dim, min_val, max_val)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Prepare data loaders
train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Training loop
n_epochs = 30
model.train()
for epoch in range(n_epochs):
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    logging.info(f"Epoch {epoch + 1}/{n_epochs}, Loss: {loss.item()}")
    


# Predict on the test set
model.eval()
with torch.no_grad():
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_pred = model(X_test_tensor).numpy()

# Convert predictions to sparse matrix
y_pred_sparse = csc_matrix(y_pred)

# Create AnnData object
adata = ad.AnnData(
    X=y_pred_sparse,
    obs=input_test_mod1.obs,
    var=input_train_mod2.var,
    uns={'dataset_id': input_train_mod1.uns['dataset_id']}
)


logging.info('Storing annotated data...')
adata.write_h5ad(par['output'], compression="gzip")
