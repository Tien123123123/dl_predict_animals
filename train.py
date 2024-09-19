import os.path
import argparse
from cnn_model import my_cnn
from animals_created import animal_dataset
from sklearn.metrics import accuracy_score, confusion_matrix
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Resize
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import shutil
from torchvision.models import resnet18, ResNet18_Weights

def plot_confusion_matrix(writer, cm, class_names, epoch):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
       cm (array, shape = [n, n]): a confusion matrix of integer classes
       class_names (array, shape = [n]): String names of the integer classes
    """

    figure = plt.figure(figsize=(20, 20))
    # color map: https://matplotlib.org/stable/gallery/color/colormap_reference.html
    plt.imshow(cm, interpolation='nearest', cmap="YlOrBr")
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    writer.add_figure('confusion_matrix', figure, epoch)

def get_args():
    parser = argparse.ArgumentParser(description="Train and Validate CNN model")

    # Path & Folder
    parser.add_argument("--data_path", type=str, default="animals", help="Dataset path")
    parser.add_argument("--save_path", type=str, default="save_models", help="Save model's parameter location")
    parser.add_argument("--log_dir", type=str, default="tensorboard", help="Tensorboard location name")

    # Hyper parameters
    parser.add_argument("--epoch", "-e", type=int, default=100, help="Total epochs for training and validating process")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for optimizer")
    parser.add_argument("--batch_size", "-b", type=int, default=16, help="Batch size for Data Loader (Train and Validation)")

    # Another
    parser.add_argument("--image_size", "-i", type=int, default=224, help="Image total size (224x224)")
    parser.add_argument("--early_stopping", type=int, default=5, help="How many time before stop training process")
    parser.add_argument("--continue_training", type=bool, default=False, help="True -> Continue training else False")

    args = parser.parse_args()
    return args

def train(args):
    size = 224
    transforms = Compose([
        ToTensor(),
        Resize((size, size))
    ])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Save models path
    board_path = args.log_dir
    save_path = args.save_path
    if not args.continue_training:
        shutil.rmtree(save_path)
        shutil.rmtree(board_path)

    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    # Tensorboard path
        # Clear tensorboard files if using new model
    if not os.path.isdir(board_path):
        os.makedirs(board_path)
    writer = SummaryWriter(args.log_dir)

    # Dataset and model
    root = args.data_path
    train_dataset = animal_dataset(root=root, is_train=True, transform=transforms)
    val_dataset = animal_dataset(root=root, is_train=False, transform=transforms)
    # model = my_cnn(num_classes=len(train_dataset.categories))
    model = resnet18(weights=ResNet18_Weights)
    model.fc = nn.Linear(in_features=512, out_features=len(train_dataset.categories), bias=True)
    model.to(device)

    # Hyper parameters
    epochs = 100
    lr = 1e-3
    batch_size = 16

    # Optim and Criterion
    optimizer = torch.optim.Adam(params= model.parameters(),lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Dataloader
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )

    # Continue Training
    if args.continue_training:
        checkpoint = torch.load(os.path.join(save_path, "last.pt"))
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"]
        best_epoch = checkpoint["best_epoch"]
        best_accuracy = checkpoint["best_accuracy"]
    else:
        start_epoch = 0
        best_epoch = 0
        best_accuracy = 0

    # Training and Validation
    number_iter = len(train_dataloader)
    for epoch in range(start_epoch, epochs):
        # Train
        model.train()
        train_loss = []
        progress_bar = tqdm(train_dataloader, colour="yellow")
        for iter, (imgs, labels) in enumerate(progress_bar):
            imgs = imgs.to(device)
            labels = labels.to(device)

            # Forward
            output = model(imgs)
            loss_val = criterion(output, labels)

            # Backward
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            train_loss.append(loss_val.item())

            # Display
            progress_bar.set_description(f"Epoch {epoch+1}/{epochs} Loss: {loss_val:0.4f}")
            writer.add_scalar("Train/Loss", np.mean(train_loss), global_step=epoch*number_iter + iter)

        # Validation
        model.eval()
        all_losses = []
        all_labels = []
        all_predictions = []
        progress_bar = tqdm(val_dataloader, colour="green")
        with torch.inference_mode():
            for imgs, labels in progress_bar:
                imgs = imgs.to(device)
                labels = labels.to(device)

                # Forward
                output = model(imgs)
                loss_val = criterion(output, labels)

                predictions = torch.argmax(output, dim=1)
                all_labels.extend(labels.tolist())
                all_predictions.extend(predictions.tolist())
                all_losses.append(loss_val.item())

            # Accuracy and Lost calculation
            accuracy = accuracy_score(all_labels, all_predictions)
            avg_loss = np.mean(all_losses)
            cf_mat = confusion_matrix(all_labels, all_predictions)
            plot_confusion_matrix(writer, cf_mat, class_names=train_dataset.categories, epoch=epoch)


            # Checkpoint
            check_point = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_epoch": best_epoch,
                "best_accuracy": best_accuracy,
                "epoch": epoch+1
            }
            # Save best accuracy
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_epoch = epoch
                torch.save(check_point, os.path.join(save_path, "best.pt"))

            # Display
            print(f"Epoch {epoch+1}/{epochs} Loss: {avg_loss:0.4f} Acc: {accuracy:0.4f}")
            writer.add_scalar("Validation/Loss", avg_loss, global_step=epoch)
            writer.add_scalar("Validation/Accuracy", accuracy, global_step=epoch)

            # Save last epoch
            torch.save(check_point, os.path.join(save_path, "last.pt"))

            # Early Stopping
            es = args.early_stopping
            if epoch - best_epoch > es:
                print(f"Train and Validate process is stopped at {epoch+1}/{epochs}")
                print(f"Best epoch {best_epoch} - Best accuracy {best_accuracy}")
                exit(0)





if __name__ == '__main__':
    args = get_args()
    train(args)