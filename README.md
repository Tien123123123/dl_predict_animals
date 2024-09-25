
Project Objective:
The goal of this project is to use PyTorch to train and validate a Convolutional Neural Network (CNN) model for classifying animals into one of 10 categories from Animal Data (Animal folder can be found on master branch):
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
  - Learning rate scheduling (StepLR) is applied for better convergence.
  - TensorBoard: Logs training metrics such as accuracy, loss, and confusion matrix.
  - Checkpointing: Saves the last trained model for continued training and the best-performing model for future predictions.
  - Argument Parser: Allows running the script with adjustable values through the terminal.
3. Inference.py
  - Functionality: This script loads the best-performing model.
  - Image Inference: It reads an image, processes it, runs the model for prediction, and displays the result.
  - Video Inference (commented out): The script supports reading frames from a video, running the model for each frame, and writing the annotated result back to an output video.
4. Results:
  - Tensorboard:
    - Train
![image](https://github.com/user-attachments/assets/f98d15f0-26b1-48c2-9db2-8285b3089ce7)

    - Validation:
![image](https://github.com/user-attachments/assets/45a7767b-e1a6-4897-946a-139a75f0a903)

  ![image](https://github.com/user-attachments/assets/d91d726b-c091-4c6f-9a0e-d7cab8054c74)
    - Confusion Matrix Img:

  ![image](https://github.com/user-attachments/assets/07eba00e-02a0-4048-8dee-802ee2c89cc4)


  - Image Output:
![image](https://github.com/user-attachments/assets/60c727c8-89d3-4674-b7aa-63fcc56bd8ad)
  - Video Output: https://github.com/user-attachments/assets/0b6c5d6d-704b-4b92-bc56-12d131fe8ef8
 
  - ![image](https://github.com/user-attachments/assets/7cea81d7-8fa4-4e2b-ba62-964e7da9fe4b)



