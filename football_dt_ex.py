from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, Compose, Resize
import cv2
import os

class FootballDataset(Dataset) :
    def __int__(self, root, isTrain):
        if isTrain:
            data_path = os.path.join(root, "train")
        else:
            data_path = os.path.join(root, "test")

    def __len__(self):
        pass
    def __getitem__(self, item):
        pass


def main() :
    transform = Compose(
        [ToTensor(),
         Resize((224, 224))]
    )
    dataset = FootballDataset(root="dataset/football", isTrain= True)


if __name__ == "__main__" :
    main()
