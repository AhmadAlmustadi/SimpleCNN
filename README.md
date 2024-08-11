# SimpleCNN
This project involves building a simple CNN model to classify images from the Fashion MNIST dataset. The model is implemented using PyTorch, and its performance can be improved by increasing the number of epochs during training.

The model consists of 3 blocks:

Block 1: A convolutional layer followed by a ReLU activation function, another convolutional layer, and then a max-pooling layer.
Block 2: Identical to Block 1.
Block 3: A flattening layer followed by a fully connected layer for classification.
Fashion MNIST Dataset
The Fashion MNIST dataset is a collection of 70,000 grayscale images, each 28x28 pixels in size, representing 10 different categories of clothing and accessories (e.g., T-shirts, trousers, bags). It serves as a more complex alternative to the classic MNIST dataset, which contains handwritten digits, and is often used as a benchmark in machine learning experiments.

Training Details
The model was trained using the following settings:

Epochs: 3
Optimizer: torch.optim.SGD with a learning rate of 0.1
Loss Function: nn.CrossEntropyLoss
Evaluation
The model's performance was evaluated using an accuracy function that was custom-written. This function measures how well the model predicts the correct labels for the images in the test set.

Suggestions for Improving the Model
Increase the Number of Epochs: The model was trained for only 3 epochs, which may not be enough for it to fully learn the patterns in the data. Increasing the number of epochs could improve the model's accuracy.

Data Augmentation: Applying techniques like random rotations, shifts, or flips to the images can help the model generalize better by providing it with a more varied dataset.

Learning Rate Adjustment: Experimenting with different learning rates or using learning rate schedules can lead to better convergence and improved performance.

Batch Normalization: Adding batch normalization layers between the convolutional and activation layers can stabilize the learning process and improve the model's accuracy.

Regularization: Implementing techniques like dropout can prevent overfitting by randomly "dropping" neurons during training, which helps the model generalize better.

Use of Pre-trained Models: Fine-tuning a pre-trained CNN model (such as ResNet or VGG) on the Fashion MNIST dataset can leverage transfer learning to improve accuracy significantly.

By applying these techniques, you can enhance the model's performance and make it more robust.
