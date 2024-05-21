import os.path
import torch
import torch.nn as nn
from dataset_1 import AnimalDataset
from model_1 import MyNeuralNetwork
from torchvision.transforms import ToTensor, Compose, Resize
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from torchvision.models import resnet50, ResNet50_Weights

def train() :
    batch_size = 16
    epochs = 10
    learning_rate = 1e-3  # 0.01
    momentum = 0.9
    log_path = "tensorboard/animals"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = Compose(
        [ToTensor(),
         Resize((224, 224))]
    )
    train_dataset = AnimalDataset(root="dataset/animals", is_train=True, transform=transform)
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=4,
        drop_last=True,
        shuffle=True
    )
    val_dataset = AnimalDataset(root="dataset/animals", is_train=False, transform=transform)
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        num_workers=4,
        drop_last=True,
        shuffle=True
    )
    num_iters = len(train_dataloader)
    if not os.path.isdir(log_path):
        os.mkdir(log_path)
    writer = SummaryWriter(log_path)
    #model = MyNeuralNetwork(num_classes=len(train_dataset.categories)).to(device)
    model = resnet50(weights=ResNet50_Weights.DEFAULT).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer =torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    for epoch in range(epochs):
        model.train()
        progress_bar_1 = tqdm(train_dataloader, colour="cyan")
        for iter, (images, labels) in enumerate(progress_bar_1):
            #foward pass
            images = images.to(device)
            labels = labels.to(device)
            predictions = model(images)
            loss = criterion(predictions, labels)

            #backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            progress_bar_1.set_description("Epoch: {}/{}. Loss:{:0.4f}".format(epoch+1, epochs, loss.item()))
            writer.add_scalar("Train/Loss", loss, epoch * num_iters + iter)
        #model validation
        model.eval()
        all_prediction = []
        all_labels = []
        all_losses = []
        Progress_bar_1 = tqdm(val_dataloader, colour="yellow")
        with torch.no_grad():
            for iter, (images, labels) in enumerate(Progress_bar_1):
                #foward pass
                images = images.to(device)
                labels = labels.to(device)
                predictions = model(images)
                loss = criterion(predictions, labels)
                predictions = torch.argmax(predictions, dim=1)
                all_labels.extend(labels.tolist())
                all_prediction.extend(predictions.tolist())
                all_losses.append(loss.item())

            acc = accuracy_score(all_labels, all_prediction)
            loss = np.mean(all_losses)
            writer.add_scalar("Val/Loss", loss, epoch)
            writer.add_scalar("Val/Accuracy", acc, epoch)
            Progress_bar_1.set_description("Epoch:{}. Val accuracy : {:0.4f}. Val loss: {:0.4f}".format(epoch+1, acc, loss))






if __name__=="__main__":
    train()