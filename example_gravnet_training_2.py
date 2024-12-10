from gravnet_model import *
import os
import pickle
import torch
from torch import optim
import phate
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

## Load data
os.chdir('/gpfs/gibbs/pi/krishnaswamy_smita/sm3299/483_project/input_data')

with open('hit_information.pickle', 'rb') as handle:
    X = pickle.load(handle)

with open('true_clusterID.pickle', 'rb') as handle:
    y = pickle.load(handle)

with open('energies.pickle', 'rb') as handle:
    E = pickle.load(handle)

with open('n_hits.pickle', 'rb') as handle:
    N = pickle.load(handle)

# Split along iz
from collections import defaultdict
i = 0
new_X = []
new_y = []
for event in X:
    labels = y[i]
    # Create a defaultdict to categorize hits by hit[5]
    hit_categories = defaultdict(list)
    label_categories = defaultdict(list)

    j = 0
    for hit in event:
        hit_value = hit[5]
        hit_categories[hit_value].append(hit)
        label_categories[hit_value].append(labels[j])
        j += 1

    # Append the hits and labels for each category (0 to 6) to new_X and new_y
    for k in range(7):  # Assuming hit[5] ranges from 0 to 6
        new_X.append(hit_categories[k])
        new_y.append(label_categories[k])

    i += 1


n_clust_to_keep = 10

true_labels = []
node_features = []
for i in range(len(y)): # i is the event
    if len(set(y[i])) >= n_clust_to_keep : # if the event has more or exactly the number of clusters
        label = torch.tensor(map_to_integers(new_y[i]), dtype=torch.int64) # this is the list of cluster assignments for event i
        this_event = torch.tensor(new_X[i], dtype = torch.float32, requires_grad = True) # this is the information for our event
        X_event = [] # this will store the information we want to keep 
        labels_event = []
        for k,j in enumerate(label): # j is the cluster label
            if j<n_clust_to_keep: # we keep the clusters with the first 10 labels
                X_event.append(this_event[k])
                labels_event.append(j)
        if X_event: 
            X_event = torch.stack(X_event)
            labels_event = torch.stack(labels_event)
            true_labels.append(labels_event)
            node_features.append(X_event)



X_train, X_test, y_train, y_test = train_test_split(node_features, true_labels, test_size=0.2, random_state=0)

train_data = []
for i in range(len(y_train)): 
    graph = Data(x = X_train[i], y = y_train[i])
    train_data.append(graph)
test_data = []
for i in range(len(y_test)): 
    graph = Data(x = X_test[i], y = y_test[i])
    test_data.append(graph)


train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
test_loader = DataLoader(test_data, batch_size=16, shuffle=False)

model = GravNetClustering()
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = torch.nn.CrossEntropyLoss()

# training loop
epochs = 400

losses = []
for epoch in range(epochs):
    print(f"Epoch {epoch}/{epochs}")
    model.train()
    epoch_loss = 0

    for batch in train_loader:
        optimizer.zero_grad()
        outputs, embed = model(batch.x) 
        loss = loss_fn(outputs, batch.y)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
    avg_loss = epoch_loss/len(train_loader)
    print(avg_loss)
    losses.append(avg_loss)


checkpoint_path = 'zfaces_k10_simple_gravnet_checkpoint.pth'
torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, checkpoint_path)

       
# testing loop
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

os.chdir('/gpfs/gibbs/pi/krishnaswamy_smita/sm3299/483_project/figures')
plt.figure(figsize=(8, 6))
plt.plot(range(1, epochs+1), losses, label='Training Loss', color='blue')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.savefig("zfaces_k10_training_curve.png",dpi=300)


total_loss = 0.0
correct_predictions = 0
total_samples = 0
embeddings = []
labels_true = []
energies = []
ev_lab = []
pred_lab = []

model.eval()
with torch.no_grad():
    for batch in test_loader:
        print(batch.x.shape)
        outputs, embeds = model(batch.x)
        print(outputs.shape)
        embeddings.append(embeds)
        loss = loss_fn(outputs, batch.y)
        total_loss += loss.item()
        energ = batch.x[:,-1] # extracts last column
        _, predicted = torch.max(outputs, 1)
        correct_predictions += (predicted == batch.y).sum().item()
        total_samples += batch.y.size(0)
        #i = 0
        
        for i,lab in enumerate(batch.y):
            ev_lab.append(lab.item())
            pred_lab.append(predicted[i].item())
            energies.append(energ[i].item())
        labels_true.append(np.array(ev_lab))
        labels_pred.append(np.array(pred_lab))
        all_energies.append(np.array(energies))
      
accuracy = 100*correct_predictions / total_samples
print("test accuracy:", accuracy)

#perform dimensionality reduction to visualize embeddings for the first event
embed = embeddings[0].detach().numpy()
labels = labels_true[0]
pred_labels = labels_pred[0] 
print(labels)
print(pred_labels)
phate_op = phate.PHATE(verbose = 0)
phate_data = phate_op.fit_transform(embed)
phate.plot.scatter2d(phate_data, figsize=(8,6), c=labels)
plt.savefig('zfaces_k10_embeddings_0_less_neighb.png', dpi=300) 

# figure of clustered event

ev_lab = np.array(ev_lab)
pred_lab = np.array(pred_lab)
energies = np.array(energies)
hom, comp, vscore = weighted_v_score(ev_lab, pred_lab,labels_weight = energies)

print("average homogeneity:",hom)
print("average completeness:", comp)
print("average weighted v-score:", vscore)
