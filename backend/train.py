import pandas as pd
import torch
import torch.nn as nn
from model import CNN
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("../data/ddos.csv")
X = df.drop("Label", axis=1).values
y = df["Label"].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
y = torch.tensor(y, dtype=torch.long)

model = CNN(X.shape[2])
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(20):
    optimizer.zero_grad()
    outputs, features = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

torch.save(model.state_dict(), "../saved_model/cnn_model.pth")
