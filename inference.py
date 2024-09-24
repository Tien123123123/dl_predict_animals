import os.path
import argparse
from sys import orig_argv

from functorch.dim import softmax
from torchvision.models.quantization import resnet18, mobilenet_v2
from cnn_model import my_cnn
import torch
import torch.nn as nn
import cv2
import numpy as np
import cv2
import warnings
warnings.filterwarnings("ignore")

def get_args():
    parser = argparse.ArgumentParser(description="Train CNN model")
    # Image
    parser.add_argument("--image-path", "-p", type=str, help="path to an image", required=True)
    parser.add_argument("--image-size", "-s", type=int, default=224, help="Common size of image")
    # Video
    parser.add_argument("--video-path", "-v", type=str, help="path to a video", required=False)
    parser.add_argument("--frame-size", "-f", type=int, default=224, help="Common size of frame")
    # Check point
    parser.add_argument("--checkpoint-path", "-c", type=str, help="Path to trained checkpoint",
                        default="save_models/resnet18/best.pt")
    args = parser.parse_args()
    return args


def inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    categories = ["butterfly", "cat", "chicken", "cow", "dog", "elephant", "horse", "sheep", "spider", "squirrel"]

    # model = mobilenet_v2(weights=None)
    # model.classifier[1] = nn.Linear(in_features=1280, out_features=len(categories), bias=True)
    model = resnet18(weights=None)
    model.fc = nn.Linear(in_features=512, out_features=len(categories), bias=True)

    model.to(device)
    checkpoint = torch.load(args.checkpoint_path)
    model.load_state_dict(checkpoint["model"])

    # For Video
    # cap = cv2.VideoCapture(args.video_path)
    # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # out_video = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*"mp4v"), int(cap.get(cv2.CAP_PROP_FPS)),
    #                             (width, height))
    # while cap.isOpened():
    #     flag, ori_frame = cap.read()
    #     if not flag:
    #         break
    #     frame = cv2.cvtColor(ori_frame, cv2.COLOR_BGR2RGB)
    #     frame = cv2.resize(frame,(args.frame_size, args.frame_size))
    #     frame = frame/225.
    #     frame = np.transpose(frame, (2,0,1))
    #     frame = frame[None,:,:,:]
    #     frame = torch.from_numpy(frame).float().to(device)
    #     softmax = nn.Softmax()
    #     with torch.no_grad():
    #         output = model(frame)
    #         probs = softmax(output[0])
    #     predicted_prob, predicted_idx = torch.max(probs, dim=0)
    #     text = f"{categories[predicted_idx]} - {predicted_prob*100:0.2f}%"
    #     ori_frame = cv2.putText(ori_frame, text, (100, 100), cv2.FONT_HERSHEY_SIMPLEX,
    #                             1, (0, 0, 255), 2, cv2.LINE_AA)
    #     out_video.write(ori_frame)
    # cap.release()
    # out_video.release()


    # For Image
    ori_image = cv2.imread(args.image_path)
    image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (args.image_size, args.image_size))
    image = image / 255.
    image = np.transpose(image, (2,0,1))
    # Convert image to 4 dims
    image = image[None,:,:,:]
    image = torch.from_numpy(image).float().to(device)
    # Evaluation
    softmax = nn.Softmax()
    model.eval()
    with torch.inference_mode():
        output = model(image)
        probs = softmax(output[0])
    predicted_prob, predicted_idx = torch.max(probs, dim=0)
    print(probs)
    print(predicted_prob, predicted_idx)
    cv2.imshow(f"{categories[predicted_idx]} - {predicted_prob*100:0.2f}%", cv2.resize(ori_image, (500, 800)))
    cv2.waitKey(0)




if __name__ == '__main__':
    args = get_args()
    inference(args)
