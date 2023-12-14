from utils import Dataset, VGG
import torch
import numpy as np
from sklearn.model_selection import train_test_split

specs = np.load('log_mel_spectrograms.npy')
labels = np.load('labels.npy')
specs = specs.reshape(-1, 64, 64, 1)
specs = (specs - np.mean(specs)) / np.std(specs)

X_train, X_temp, y_train, y_temp = train_test_split(specs, labels, test_size=0.3, random_state=40)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.67, random_state=40)
print("There are " + str(len(X_train)) + " training samples, " + str(len(X_val)) + " validation samples, and " + str(len(X_test)) + " testing samples.")

test_dataset = Dataset(X_test, y_test)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=True)

vgg = VGG()
vgg.load_state_dict(torch.load('./Metrics/model_final_30.ckpt'))
vgg.eval()  

class_labels = ['cel', 'cla', 'flu', 'gac', 'gel', 'org', 'pia', 'sax', 'tru', 'vio', 'voi']

for _ in range(5):
    images, labels = next(iter(test_loader))
    images = images.float()
    labels = labels.long()

    with torch.no_grad():
        outputs = vgg(images)
        _, predicted = torch.max(outputs.data, 1)

    true_label = int(labels.numpy())
    predicted_label = int(predicted.numpy())

    print(f'True Class: {class_labels[true_label]}, Predicted Class: {class_labels[predicted_label]}')
