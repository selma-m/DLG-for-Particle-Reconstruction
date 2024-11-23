from gravnet_model import *
import os
import pickle
import torch
from torch import optim
import torch.nn.functional as F


## Load data
os.chdir('/input_data')

with open('hit_information.pickle', 'rb') as handle:
    X = pickle.load(handle)

with open('true_clusterID.pickle', 'rb') as handle:
    y = pickle.load(handle)

with open('energies.pickle', 'rb') as handle:
    E = pickle.load(handle)

with open('n_hits.pickle', 'rb') as handle:
    N = pickle.load(handle)



# Initialize the model and optimizer
model = GravNetClustering()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 100
checkpoint_interval = 3  # Save checkpoint every 10 epochs
os.chdir('/gpfs/gibbs/pi/krishnaswamy_smita/sm3299/483_project/checkpoints')
checkpoint_path = 'gravnet_checkpoint.pth'

# Try loading from checkpoint
try:
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    print(f"Resumed training from epoch {start_epoch}")
except FileNotFoundError:
    start_epoch = 0
    print("No checkpoint found, starting from scratch.")

# Training loop
for epoch in range(start_epoch, epochs):
    print(f"Epoch {epoch}/{epochs}")
    model.train()
    
    epoch_loss = 0
    for i in range(650):  # Adjust batch iteration as needed
        true_labels = torch.tensor(map_to_integers(y[i]), dtype=torch.int64)
        input_data = torch.tensor(X[i], dtype=torch.float32, requires_grad = True)

        outputs, embed = model(input_data) 
 
        loss = weighted_soft_v_measure(true_labels,outputs,labels_weight = E[i])
        epoch_loss += loss
        loss.backward()
        optimizer.step()
    epoch_loss = epoch_loss / 650

    # Checkpoint saving
    if epoch % checkpoint_interval == 0:
        torch.save({
            'epoch': epoch + 1,  # Save the epoch for resumption
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, checkpoint_path)
        print(f"Checkpoint saved at epoch {epoch + 1}, loss: {epoch_loss}")

# Final model saving after training
torch.save(model.state_dict(), 'trained_gravnet_kl.pth')
