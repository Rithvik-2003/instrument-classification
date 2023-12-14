from utils import Dataset, VGG

import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.metrics import precision_recall_fscore_support

specs = np.load('log_mel_spectrograms.npy')
labels = np.load('labels.npy')
specs = specs.reshape(-1, 64, 64, 1)
specs = (specs - np.mean(specs)) / np.std(specs)

X_train, X_temp, y_train, y_temp = train_test_split(specs, labels, test_size=0.3, random_state=40)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.67, random_state=40)
print("There are " + str(len(X_train)) + " training samples, " + str(len(X_val)) + " validation samples, and " + str(len(X_test)) + " testing samples.")

test_dataset = Dataset(X_test, y_test)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=16, shuffle=False)
all_true_labels = []
all_predicted_labels = []

vgg = VGG()
vgg.load_state_dict(torch.load('./Metrics/model_final_30.ckpt'))
vgg.eval()  
device = torch.device('cpu')
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        
        images = images.to(device).float()
        labels = labels.to(device).long()
        outputs = vgg(images)
        
        _, predicted = torch.max(outputs.data, 1)
        all_true_labels.extend(labels.cpu().numpy())
        all_predicted_labels.extend(predicted.cpu().numpy())
        

confusion_mat = confusion_matrix(all_true_labels, all_predicted_labels)
class_labels = ['cel', 'cla', 'flu', 'gac', 'gel', 'org', 'pia', 'sax', 'tru', 'vio', 'voi']
accuracy = accuracy_score(all_true_labels, all_predicted_labels)
precision, recall, f1, _ = precision_recall_fscore_support(all_true_labels, all_predicted_labels, average='weighted')
micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(all_true_labels, all_predicted_labels, average='micro')
macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(all_true_labels, all_predicted_labels, average='macro')
class_precisions, class_recalls, class_f1s, _ = precision_recall_fscore_support(all_true_labels, all_predicted_labels, labels=np.unique(all_true_labels), average=None)

for i, class_label in enumerate(class_labels):
    print(f"Class {class_label} - Precision: {class_precisions[i]}, Recall: {class_recalls[i]}, F1-score: {class_f1s[i]}")

print("Accuracy:", accuracy)
print("Weighted Precision:", precision)
print("Weighted Recall:", recall)
print("Weighted F1 Score:", f1)
print("Micro Precision:", micro_precision)
print("Micro Recall:", micro_recall)
print("Micro F1 Score:", micro_f1)
print("Macro Precision:", macro_precision)
print("Macro Recall:", macro_precision)
print("Macro F1 Score:", macro_f1)
print("Confusion Matrix:")
print(confusion_mat)
normalized_confusion_mat = normalize(confusion_mat, axis=1, norm='l1')

# plt.figure(figsize=(10, 8))
# sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
# plt.title('Confusion Matrix')
# plt.xlabel('Predicted Labels')
# plt.ylabel('True Labels')
# plt.savefig(f"./Metrics/new_matrix_final_30.png")

sns.set(style="darkgrid")
dark_palette = sns.dark_palette("red", as_cmap=True)
sns.heatmap(normalized_confusion_mat, annot=True, fmt='.3f', cmap=dark_palette, xticklabels=class_labels, yticklabels=class_labels, cbar=False)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.savefig(f"./Metrics/new_norm_matrix_final_30.png")
