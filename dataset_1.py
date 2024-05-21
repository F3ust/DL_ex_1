import torch.cuda
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, Compose, Resize
import os
import cv2
import torch.nn as nn
from PIL import Image
import pickle

class AnimalDataset(Dataset):
    def __init__(self, root, is_train, transform):
        if is_train:
            data_path = os.path.join(root, "train")
        else:
            data_path = os.path.join(root, "test")
        self.all_images_path = []
        self.all_labels =[]
        self.categories =["butterfly", "cat", "chicken", "cow", "dog", "horse", "spider", "elephant", "sheep", "squirrel"]
        for index, category in enumerate(self.categories):
            category_path = os.path.join(data_path, category)
            for item in os.listdir(category_path):
                image_path = os.path.join(category_path, item)
                self.all_images_path.append(image_path)
                self.all_labels.append(index)
        self.transform = transform

    def __len__(self):
        return len(self.all_labels)

    def __getitem__(self, item):
        image_path = self.all_images_path[item]
        image = cv2.imread(image_path)
        if self.transform:
            image = self.transform(image)
        label = self.all_labels[item]
        return image, label

def main():
    transform = Compose(
        [ToTensor(),
         Resize((224, 224))]
    )

    dataset = AnimalDataset(root="dataset/animals", is_train=True, transform=transform)
    # image, label = dataset[1213]
    # print(label)
    # cv2.imshow(str(label), image)
    # cv2.waitKey(0)

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=4,
        num_workers=4,
        drop_last=True,
        shuffle=True,
    )



if __name__ == "__main__" :
    main()