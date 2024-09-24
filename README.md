# dl_predict_animals

Project Objective:
The goal of this project is to use PyTorch to train and validate a Convolutional Neural Network (CNN) model for classifying animals into one of 10 categories: 
 1.butterfly
 2.cat
 3.chicken
 4.cow
 5.dog
 6.elephant
 7.horse
 8.sheep
 9.spider
 10.squirrel

 animals_created.py
  - animals_created.py will take data from animals file to create dataset for training with CNN model
  - Image Loading: Images are loaded using PIL and transformed via ToTensor() and Resize() to a uniform size of 224x224 pixels.
  - Categories and Labels: Each subfolder name represents a class. Labels are automatically assigned by iterating through these subfolders.
  - PyTorch Integration: The dataset integrates seamlessly into PyTorch's DataLoader, making it ready for model training and evaluation.

 Train.py:
  1. Model Selection: The script supports training using three different models:
    - A custom CNN model (my_cnn).
    - ResNet18 pre-trained on ImageNet.
    - MobileNetV2, also pre-trained on ImageNet.
  2. Training Process:
    - Optimized using the Adam optimizer and cross-entropy loss.
    - Learning rate scheduling (StepLR) is applied for better convergence.
    - TensorBoard: Logs training metrics (accuracy, loss, confusion matrix)
    - Checkpoint: saves the last trained model for continue training and best-performing model for predicting.
  4. Argument Parser: Allowing running file with adjusted values through terminal

Inference.py:
  - Load best-performing model
  - Image Inference: Reads an image, processes it, runs the model for prediction and displays the result.
  - Video Inference (commented): The script supports reading frames from a video, running the model for each frame and writing the annotated result back to an output video.

Result:
Image: 
![image](https://github.com/user-attachments/assets/60c727c8-89d3-4674-b7aa-63fcc56bd8ad)
Video:
![image](https://github.com/user-attachments/assets/100ff5d6-998a-4679-a898-4688fa7a0a3e)

