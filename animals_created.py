import os
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor, Resize
from PIL import Image

class animal_dataset(Dataset):
    def __init__(self, root, is_train, transform):
        self.root = root
        self.categories = []
        self.imgs = []
        self.labels = []
        with os.scandir(root) as directory:
            for item in directory:
                if item.is_dir():
                    classes_dir = os.path.join(root, "train" if is_train==True else "test")

        with os.scandir(classes_dir) as directory:
            for item in directory:
                self.categories.append(item.name if item.is_dir() else None)

        for idx, animal in enumerate(self.categories):
            animal_dir = os.path.join(classes_dir, animal)
            with os.scandir(animal_dir) as directory:
                for item in directory:
                    if item.is_file():
                        self.imgs.append(os.path.join(animal_dir, item.name))
                        self.labels.append(idx)

        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        # image = cv2.imread(self.imgs[index])
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.open(self.imgs[index]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = self.labels[index]

        return image, label

if __name__ == '__main__':
    transform = Compose([
        ToTensor(),
        Resize((224,224))
    ])
    root = 'animals'
    dataset = animal_dataset(root=root, is_train=True, transform=transform)
    print(dataset.__len__())
    image, label = dataset[23000]

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=16,
        shuffle=True,
        drop_last=True
    )

    for image, label in dataloader:
        print(image.shape)
        print(label)

    # cv2.imshow(label, image)
    # cv2.waitKey(0)