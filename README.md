Project: dl_predict_animals
Project Objective:
The goal of this project is to use PyTorch to train and validate a Convolutional Neural Network (CNN) model for classifying animals into one of 10 categories:
- Butterfly
- Cat
- Chicken
- Cow
- Dog
- Elephant
- Horse
- Sheep
- Spider
- Squirrel
  Project Components:
1. animals_created.py
- Functionality: This script creates a dataset for training the CNN model from the images in the specified animals folder.
- Image Loading: Images are loaded using the PIL library and transformed via ToTensor() and Resize() to a uniform size of 224x224 pixels.
- Categories and Labels: Each subfolder name represents a class. Labels are automatically assigned by iterating through these subfolders.
- PyTorch Integration: The dataset integrates seamlessly into PyTorch's DataLoader, making it ready for model training and evaluation.
2. Train.py
- Model Selection: This script supports training using three different models:
  - A custom CNN model (my_cnn).
  - ResNet18 pre-trained on ImageNet.
  - MobileNetV2, also pre-trained on ImageNet.
- Training Process:
  - Optimized using the Adam optimizer and cross-entropy loss.
  -Learning rate scheduling (StepLR) is applied for better convergence.
  -TensorBoard: Logs training metrics such as accuracy, loss, and confusion matrix.
  -Checkpointing: Saves the last trained model for continued training and the best-performing model for future predictions.
  -Argument Parser: Allows running the script with adjustable values through the terminal.
3. Inference.py
  -Functionality: This script loads the best-performing model.
  -Image Inference: It reads an image, processes it, runs the model for prediction, and displays the result.
  -Video Inference (commented out): The script supports reading frames from a video, running the model for each frame, and writing the annotated result back to an output video.
Results:
Image Output:
![image](https://github.com/user-attachments/assets/60c727c8-89d3-4674-b7aa-63fcc56bd8ad)
Video Output:
![image](https://github.com/user-attachments/assets/100ff5d6-998a-4679-a898-4688fa7a0a3e)

