import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from utils import Dataset, VGG
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
    
specs = np.load('log_mel_spectrograms.npy')
labels = np.load('labels.npy')

specs = specs.reshape(-1, 64, 64, 1)
specs = (specs - np.mean(specs)) / np.std(specs)

X_train, X_temp, y_train, y_temp = train_test_split(specs, labels, test_size=0.3, random_state=40)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.67, random_state=40)
print("There are " + str(len(X_train)) + " training samples, " + str(len(X_val)) + " validation samples, and " + str(len(X_test)) + " testing samples.")

train_dataset = Dataset(X_train, y_train)
val_dataset = Dataset(X_val, y_val)
test_dataset = Dataset(X_test, y_test)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=16, shuffle=False)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=16, shuffle=False)

vgg = VGG()
print(vgg)
criterion = nn.CrossEntropyLoss()

learning_rate = 0.001
optimizer = torch.optim.Adam(vgg.parameters(), lr = learning_rate)

device = torch.device('cpu')
num_epochs = 30

training_loss_list = []
validation_loss_list = []
training_accuracy_list = []
validation_accuracy_list = []
best_val_accuracy = 0.0

for epoch in range(num_epochs):
    vgg.train()
    total_loss = 0
    correct = 0
    total_samples = 0

    for images, labels in tqdm(train_loader, desc=f'Training Epoch {epoch + 1}/{num_epochs}'):

        images = images.to(device).float()
        labels = labels.to(device).long()

        outputs = vgg(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total_samples += labels.size(0)
        correct += (predicted == labels).sum().item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_loss = total_loss / len(train_loader)
    training_loss_list.append(avg_loss)
    training_accuracy = correct / total_samples * 100
    training_accuracy_list.append(training_accuracy)
    print(f'Training Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {training_accuracy:.2f}%')

    vgg.eval()
    val_loss = 0
    val_correct = 0
    val_total_samples = 0

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f'Validation Epoch {epoch + 1}/{num_epochs}'):

            images = images.to(device).float()
            labels = labels.to(device).long()

            outputs = vgg(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            val_total_samples += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    avg_val_loss = val_loss / len(val_loader)
    validation_loss_list.append(avg_val_loss)
    val_accuracy = val_correct / val_total_samples * 100
    validation_accuracy_list.append(val_accuracy)
    print(f'Validation Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.2f}%')

    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(vgg.state_dict(), f'./Metrics/model_final_{num_epochs}.ckpt')

plt.figure(figsize=(12, 8))
epochs = list(range(1, num_epochs + 1))
plt.plot(epochs, training_loss_list, label='Training Loss')
plt.plot(epochs, validation_loss_list, label='Validation Loss')
plt.title("Training and Validation Metrics vs Epochs")
plt.xlabel("Epoch")
plt.legend()
plt.savefig(f"./Metrics/loss_metrics_final_{num_epochs}.png")

plt.plot(epochs, training_accuracy_list, label='Training Accuracy')
plt.plot(epochs, validation_accuracy_list, label='Validation Accuracy')
plt.title("Training and Validation Metrics vs Epochs")
plt.xlabel("Epoch")
plt.legend()
plt.savefig(f"./Metrics/accuracy_metrics_final_{num_epochs}.png")
