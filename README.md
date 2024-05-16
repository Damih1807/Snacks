---
pretty_name: Snacks
task_categories:
- image-classification
- computer-vision
license: cc-by-4.0
---

# Dataset Card for Snacks

## Dataset Summary

This is a dataset of 20 different types of snack foods that accompanies the book [Machine Learning by Tutorials](https://www.raywenderlich.com/books/machine-learning-by-tutorials/v2.0).

The images were taken from the [Google Open Images dataset](https://storage.googleapis.com/openimages/web/index.html), release 2017_11. 

## Dataset Structure

Number of images in the train/validation/test splits:

```nohighlight
train    4838
val      955
test     952
total    6745
```

Total images in each category:

```nohighlight
apple         350
banana        350
cake          349
candy         349
carrot        349
cookie        349
doughnut      350
grape         350
hot dog       350
ice cream     350
juice         350
muffin        348
orange        349
pineapple     340
popcorn       260
pretzel       204
salad         350
strawberry    348
waffle        350
watermelon    350
```

To save space in the download, the images were resized so that their smallest side is 256 pixels. All EXIF information was removed.

### Data Splits

Train, Test, Validation

## Licensing Information

Just like the images from Google Open Images, the snacks dataset is licensed under the terms of the Creative Commons license. 

The images are listed as having a [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/) license. 

The annotations are licensed by Google Inc. under a [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) license. 

The **credits.csv** file contains the original URL, author information and license for each image.

# PROCESSING
Image Classification with ViT and ResNet18
This project demonstrates how to preprocess data, perform feature engineering, and build image classification models using Vision Transformer (ViT) and ResNet18. The models are trained on a dataset from Hugging Face and evaluated for performance.

## Installation
First, you need to install the required libraries. You can do this using pip:
pip install renumics-spotlight sliceguard[all] scikit-learn transformers datasets

## Data Loading
We use the sliceguard library to read a dataset from the Hugging Face website. Here’s a basic example of how to load the data:
from datasets import load_dataset

# Load the dataset from Hugging Face
dataset = load_dataset("Matthijs/snacks")

# Split the dataset into training and test sets
train_dataset = dataset['train']
test_dataset = dataset['test']

## Feature Engineering
We perform the following feature engineering steps:

*Label Encoding*: Encode the output column.
*Image Transformation*: Transform the image column.
## Model Training
We train both the Vision Transformer (ViT) and ResNet18 models with the same number of epochs and batch size.

**Vision Transformer (ViT)**
from transformers import ViTForImageClassification, TrainingArguments, Trainer

# Define the model
model_vit = ViTForImageClassification.from_pretrained(
    'google/vit-base-patch16-224-in21k',
    num_labels=len(label_encoder.classes_)
)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=10,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir='./logs',
)

# Define Trainer
trainer_vit = Trainer(
    model=model_vit,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# Train the model
trainer_vit.train()
**ResNet18**
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18
from torch.utils.data import DataLoader, Dataset

class CustomDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['pixel_values']
        label = item['label']
        return image, label

# Create data loaders
train_loader = DataLoader(CustomDataset(train_dataset), batch_size=16, shuffle=True)
test_loader = DataLoader(CustomDataset(test_dataset), batch_size=16, shuffle=False)

# Define the model
model_resnet = resnet18(pretrained=True)
model_resnet.fc = nn.Linear(model_resnet.fc.in_features, len(label_encoder.classes_))

# Training settings
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_resnet.parameters(), lr=0.001)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_resnet.to(device)

# Train the model
num_epochs = 10
for epoch in range(num_epochs):
    model_resnet.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model_resnet(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}")

# Evaluate the model
model_resnet.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model_resnet(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy of the model on the test images: {100 * correct / total}%")


