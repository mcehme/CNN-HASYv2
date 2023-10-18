# MNIST CNN Classifier

This Python script demonstrates training and evaluating a Convolutional Neural Network (CNN) on the MNIST dataset for handwritten digit recognition.

## Sections

### 1. Data Handler
- Handles loading and preprocessing of the MNIST dataset.

### 2. Model
- Defines the architecture of the CNN model.

### 3. Runner
- Utilizes k-fold cross-validation to evaluate the model's performance.

### 4. Trainer
- Summarizes and visualizes the model's training performance.

## Instructions

1. Make sure you have the required libraries installed. You can install them using the following command:
    ```
    pip install numpy matplotlib tensorflow scikit-learn
    ```

2. Execute the script `mnist_cnn_classifier.py`.

## Sections Details

### 1. Data Handler
- `load_dataset()`: Loads the MNIST dataset, reshapes it, and performs one-hot encoding of target values.
- `prep_pixels(train, test)`: Converts pixel values to floats and normalizes them to the range of 0-1.

### 2. Model
- `define_model()`: Defines the CNN model with convolutional layers, max-pooling layers, dense layers, and softmax output.

### 3. Runner
- `evaluate_model(dataX, dataY, n_folds=5)`: Evaluates the model using k-fold cross-validation.

### 4. Trainer
- `summarize_diagnostics(histories)`: Plots training and validation loss, as well as training and validation accuracy.
- `summarize_performance(scores)`: Prints mean accuracy and standard deviation, and displays a box and whisker plot of results.
- `run_test_harness()`: Orchestrates the entire process: loading data, training the model, and summarizing results.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Remember to include a `LICENSE` file if you want to specify the licensing terms for your code. If you choose a different license, make sure to adjust the content of the `LICENSE` section accordingly.
