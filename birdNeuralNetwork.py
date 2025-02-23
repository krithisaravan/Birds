import pandas as pd
import numpy as np
import image
import shutil
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms
#from readData import CustomBirdDataset

path_to_train_images = 'C:/Users/greap/Downloads/dataorbit/files/Birds/train/'
path_to_test_images = 'C:/Users/greap/Downloads/dataorbit/files/Birds/test/'
train_label_file = 'train_labels.csv'
test_label_file = 'test_labels.csv'

# restructure data
test_csv = pd.read_csv('C:/Users/greap/Downloads/dataorbit/files/Birds/test.csv')
train_csv = pd.read_csv('C:/Users/greap/Downloads/dataorbit/files/Birds/train.csv')

print(test_csv.head())
print(train_csv.head())

train_df = pd.DataFrame(train_csv)
test_df = pd.DataFrame(test_csv)
#class_distr = train_df['target'].value_counts()
#sorted_class_distr = class_distr.sort_index()
#print("Class distribution:\n", class_distr)
#total_samples = len(train_df)

#testing and training data sorted into class folders
sorted_test_images_path = 'C:/Users/greap/Downloads/dataorbit/files/Birds/sorted_test'
if not os.path.exists(sorted_test_images_path):
    os.makedirs(sorted_test_images_path)

for target_value in test_csv['target'].unique():
    class_folder_path = os.path.join(sorted_test_images_path, str(target_value))
    if not os.path.exists(class_folder_path):
        os.makedirs(class_folder_path)

for index, row in test_csv.iterrows():
    image_name = row['img_id']
    target_value = row['target']

    source_image_path = os.path.join(path_to_test_images, image_name)
    destination_image_path = os.path.join(sorted_test_images_path, str(target_value), image_name)

    if os.path.exists(source_image_path):
        shutil.copy(source_image_path, destination_image_path)  # Copy instead of move
    else:
        print(f"Source image not found: {source_image_path}")

sorted_train_images_path = 'C:/Users/greap/Downloads/dataorbit/files/Birds/sorted_train'
if not os.path.exists(sorted_train_images_path):
    os.makedirs(sorted_train_images_path)

for target_value in train_csv['target'].unique():
    class_folder_path = os.path.join(sorted_train_images_path, str(target_value))
    if not os.path.exists(class_folder_path):
        os.makedirs(class_folder_path)
for index, row in train_csv.iterrows():
    image_name = row['img_id']
    target_value = row['target']

    source_image_path = os.path.join(path_to_train_images, image_name)
    destination_image_path = os.path.join(sorted_train_images_path, str(target_value), image_name)

    if os.path.exists(source_image_path):
        shutil.copy(source_image_path, destination_image_path)  # Copy instead of move
    else:
        print(f"Source image not found: {source_image_path}")


transform_custom = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((.5,.5,.5), (.5, .5, .5))
])

# image folder can only work if the directory is sorted such that the images are in within folders
# of their own class

#train_data = torchvision.datasets.ImageFolder(root = path_to_train_images, transform = transform_custom)
train_data = torchvision.datasets.ImageFolder(root = sorted_train_images_path, transform = transform_custom)
#test_data = torchvision.datasets.ImageFolder(root = path_to_test_images, transform = transform_custom)
test_data = torchvision.datasets.ImageFolder(root = sorted_test_images_path, transform = transform_custom)

train_loader = torch.utils.data.DataLoader(train_data, batch_size = 32, shuffle = True, num_workers = 4)
test_loader = torch.utils.data.DataLoader(train_data, batch_size = 32, shuffle = True, num_workers = 4)

image, label = train_data[0]
image.size() # torch.Size[x , y , z]

class_names = range(0, train_df['target'].nunique()) # (0, 50)
input_channels = train_df['target'].nunique()

class NeuralNet(nn.Module): # base class for a neural network in PyTorch
    def __init__(self): # defines feed-forward logic structure
        super().__init__()
        # 1st convulution layer
        # specify input (kernel/filter) - 3 parameters from (image.size())
        ## theory: moves over image
        ## filter: completely random at first: then filter trains overtime with feature maps
        # specify output (feature map) -  extracts "features" by dot product vector operations

        # takes in x input channels, (free) # of feature maps, (free) kernelsize (matrix size of filter)
        self.convLayer1 = nn.Conv2d(input_channels, 12, 5)
        # y - 5 = inputSize - kernelSize = 50 - 5 = 45
        # stride is the parameter that scales shifted pixels of the filter
        # let stride be unitary = 1

        # OUTPUT #
        # new shape post-convulution: image.shape() = (12, y-5, y-5) = (12, 23, 23)

        self.pool = nn.MaxPool2d(2,2) # creates a pooling layer with 2x2 portions
        # based on max() criteria, most important rgb value will be extracted from pool
        # compresses dimensions even further

        # new shape post-pooling: image.shape() = (12, (y-5)/2, (y-5)/2)

        # takes 12 input channels, scale up feature map size by 2 (2x2 max pool)
        self.conv2 = (12, 24, 5) # (24, 19, 19)
        # (y-5)/2 - 5 + 1 = y2 = 23 - 5 + 1 = 19

        # OUTPUT # y2/2 rounds to 10
        # new shape post-convulution: image.shape() = (24, y2/2, y2/2) = (24, 10, 10)


        # begin to flatten shape: Flatten (24 * y2/2 * y2/2), let (*) = o = 2400
        self.fc1 = nn.Linear(2400, 240) #(o, o / (y2/2))
        self.fc2 = nn.Linear(240, 90) # 90 is arbitrary but bounded to make smaller layer dim
        # fc2 : (o / (y2/2), 90)
        self.fc3 = nn.Linear(90, 50) # 50 class output


    def forward(self, x):
        """
        x - the input image to be dissected
        """

        x = self.pool(F.relu(self.conv1(x))) # breaks linearity, then fed into pooling layer
        x = self.pool(F.relu(self.conv2(x))) # again
        x = torch.flatten(x, 1) # flattening operation
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        # random/intialized deterministically at beginning,
        # then trained into neural network by showing image examples
        # model begins learning

        return x

# run it

model = NeuralNet()
loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.paramters(), lr = .001, momentum = .9)
# helps us to minimize an error function(loss function)


# accuracy (for evaluation)
def accuracy(y_pred, y_true):
    _, predicted = torch.max(y_pred, 1)
    correct = (predicted == y_true).sum().item()
    total = y_true.size(0)
    return correct / total


for epoch in range(64): # optimal training sessions
    print(f'Training epoch {epoch}...')

    running_loss = 0.0

    for i, data in enumerate(train_loader):
        inputs, labels = data

        optimizer.zero_grad()

        outputs = model(inputs) # last stage of neural network

        loss = loss_function(outputs, labels) # labels are desired outputs

        loss.backward() # gradient computation via back propagation (pytorch)

        optimizer.step() # learning rate step into the "correct direction" of params

        running_loss += loss.item()


    print(f'Loss: {running_loss / len(train_loader):.4f}')

    # evaluation

    model.eval()
    correct_pred = 0
    total_pred = 0

    with torch.no_grad():
      for inputs, labels in test_loader:
        outputs = model(inputs)
        batch_accuracy = accuracy (outputs, labels)
        correct_pred += (outputs.argmax(1) == labels).sum().item()
        total_pred += labels.size(0)

    test_accuracy = correct_pred/total_pred
    print(f'Epoch {epoch} test accuracy: {test_accuracy:.5f}')


# softmax activation layer

prediction = model_softmax(data.x)
_, y_pred = pred_model.max(1) # max value with respect to axis one

print("Model outputs", y_pred)

# check accuracy after predictions if desired (realtime updates)
correct_guess = (data.y == y_pred).sum().item()
model_accuracy = correct_guess / len(data)
print("model accuracy: ", model_accuracy)

