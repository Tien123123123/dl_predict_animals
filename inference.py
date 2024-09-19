import os.path
import argparse
from torchvision.models.quantization import resnet18
from cnn_model import my_cnn
from animals_created import animal_dataset
import torch
import cv2
import numpy as np

def get_args():
    parser = argparse.ArgumentParser(description="Train CNN model")
    parser.add_argument("--image-path", "-p", type=str, default="Image/01.jpg", help="path to an image", required=True)
    parser.add_argument("--image-size", "-s", type=int, default=224, help="Common size of image")
    parser.add_argument("--checkpoint-path", "-c", type=str, help="Path to trained checkpoint",
                        default="save_models/best.pt")
    args = parser.parse_args()
    return args


def inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    categories = ["butterfly", "cat", "chicken", "cow", "dog", "elephant", "horse", "sheep", "spider",
                  "squirrel"]

    model = resnet18(weights=None)
    num_features = model.fc.in_features
    model.fc = torch.nn.Linear(num_features, len(categories))

    model.to(device)
    checkpoint = torch.load(args.checkpoint_path)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    ori_image = cv2.imread(args.image_path)
    image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (args.image_size, args.image_size))
    image = image / 255.0
    image = np.transpose(image, (2, 0, 1))
    image = image[None, :, :, :]
    image = torch.from_numpy(image).float().to(device)

    softmax = torch.nn.Softmax(dim=1)
    with torch.no_grad():
        output = model(image)
        probs = softmax(output)

    predicted_prob, predicted_idx = torch.max(probs, dim=1)
    cv2.imshow("{}: {:0.2f} %".format(categories[predicted_idx.item()], predicted_prob.item() * 100), ori_image)
    cv2.waitKey(0)


if __name__ == '__main__':
    args = get_args()
    inference(args)