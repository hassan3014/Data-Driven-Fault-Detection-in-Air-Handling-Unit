# Data Driven Fault Detection in Air Handling Unit (AHU)
This project implements a fault detection system for Heating, Ventilation, and Air Conditioning (HVAC) systems using machine learning models. Three different models are trained and deployed: a deep neural network (DNN), a k-nearest neighbors (KNN) classifier, and a support vector machine (SVM) classifier. The models are trained on operational data from HVAC units to predict faults based on various features.

## Requirements
- pandas
- numpy
- scikit-learn
- matplotlib
- tensorflow
- Django

## Usage
1. Clone the repository to your local machine: 
    ```
    //github.com/hassan3014/Data-Driven-Fault-Detection-in-Air-Handling-Unit.git
    ```

2. Ensure you have the required Python libraries installed by running: 
    ```
    pip install -r requirements.txt
    ```

3. Configure Django settings in `settings.py`, including the `STATICFILES_DIRS` setting to point to the directory containing the dataset.

4. Run the Django application to train the models and serve predictions: 
    ```
    python manage.py runserver
    ```

## Models

### Deep Neural Network (DNN)
- Trained using TensorFlow.
- Architecture: Input layer (Flatten) -> Hidden layers (Dense) -> Output layer (Dense).
- Activation function: ReLU for hidden layers, Sigmoid for output layer.
- Loss function: Binary cross-entropy.
- Optimizer: Adam.

### K-Nearest Neighbors (KNN) Classifier
- Trained using scikit-learn.
- Utilizes the k-nearest neighbors algorithm.
- Default number of neighbors: 5.

### Support Vector Machine (SVM) Classifier
- Trained using scikit-learn.
- Utilizes the linear kernel.

## Evaluation
Models are evaluated using metrics such as accuracy and confusion matrix. The confusion matrix helps assess the models' performance in terms of true positives, true negatives, false positives, and false negatives.

## Files
- `deep_NN.py`: Python script containing the code for training the DNN model.
- `knn.py`: Python script containing the code for training the KNN classifier.
- `svm.py`: Python script containing the code for training the SVM classifier.
- `settings.py`: Django settings file for configuration.
- `views.py`: Django views file containing the logic for rendering templates and serving predictions.
- `urls.py`: Django URL configurations.
- `wsgi.py`: Django WSGI config for deployment.
- `asgi.py`: Django ASGI config for asynchronous servers.
- `manage.py`: Django command-line utility for administrative tasks.

## Acknowledgements
The dataset used in this project was obtained from [Pacific Lab Northwest](insert_dataset_link).

## Contact
For any inquiries or support, please contact [hassan.malik1574@gmail.com].
