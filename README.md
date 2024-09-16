# Brain-Tumor-Cassification-ViT

## Overview
This project focuses on the detection and classification of brain tumors using MRI images. By leveraging the Vision Transformer (ViT) architecture, the system efficiently processes medical images, classifying them into different types of brain tumors with high accuracy. This solution aims to assist medical professionals in diagnosing brain tumors quickly and accurately.

## Project Features
- **Tumor Detection**: Automatically identifies whether a brain tumor is present in an MRI scan.
- **Tumor Classification**: Classifies detected tumors into types, such as Glioma, Meningioma, Pituitary tumors, etc.
- **Vision Transformer (ViT)**: Utilizes transformer-based architecture for vision tasks, enhancing model performance by capturing complex patterns in MRI images.
- **High Performance**: Achieves improved accuracy over traditional Convolutional Neural Networks (CNNs) by utilizing transformer-based attention mechanisms.

## Dataset
The model is trained on publicly available brain MRI datasets, which consist of both labeled tumor and non-tumor images. Each image is annotated with the presence of a tumor and its type (if present).

### Dataset Source:
- The dataset used for this project can be sourced from open medical image repositories, such as Kaggle, which provide MRI scans with labeled tumor information.
- You will need to upload the dataset into the notebook in Google Colab or link it from your Google Drive.

## Requirements

### Software and Libraries:
- **Python 3.x**: Programming language used for model development.
- **Google Colab**: Recommended for leveraging GPU support during model training.
- **TensorFlow or PyTorch**: Deep learning frameworks used to build and train the Vision Transformer model.
- **Vision Transformer (ViT)**: A transformer-based model designed for image classification tasks.
- **NumPy**: For numerical operations.
- **Matplotlib & Seaborn**: For data visualization and plotting results.
- **Scikit-learn**: For performance metrics like confusion matrix, accuracy, and classification reports.
- **OpenCV**: For image preprocessing, such as resizing and normalization.

### Installation:
To install the necessary libraries, run the following commands in your Colab environment:

!pip install tensorflow matplotlib numpy opencv-python scikit-learn

## Model Architecture

### Vision Transformer (ViT):
The Vision Transformer architecture is a recent advancement in deep learning that utilizes the transformer architecture, originally designed for NLP tasks, for image classification. It splits an input image into patches, treats them as sequence data, and applies self-attention mechanisms to learn spatial relationships in the image.

**Key Components:**
1. **Patch Embedding**: Input images are divided into patches and embedded into a lower-dimensional space.
2. **Position Encoding**: Each patch is associated with positional information, enabling the model to maintain spatial understanding.
3. **Self-Attention Layers**: Transformer layers apply self-attention mechanisms to learn relationships between different parts of the image.
4. **Classification Head**: A fully connected layer at the end classifies the input image into tumor types.

## Training the Model

### Data Preprocessing:
1. Load the MRI images from the dataset.
2. Normalize the image pixel values between 0 and 1.
3. Resize the images to a fixed input size (typically 224x224).
4. Split the data into training, validation, and test sets.

### Model Training:
1. Initialize the Vision Transformer model using TensorFlow or PyTorch.
2. Compile the model with a loss function, optimizer, and evaluation metrics. For classification, you can use `Categorical Crossentropy` and `Adam` optimizer.
3. Train the model using the training set while validating on the validation set.
4. Use techniques like data augmentation and dropout to prevent overfitting.

```python
# Example: Compiling the ViT model in TensorFlow
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)
```

### Evaluation:
Evaluate the model on the test set using metrics like:
- **Accuracy**
- **Precision**
- **Recall**
- **F1-score**
- **Confusion Matrix**

## Usage Instructions

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/brain-tumor-classification.git
   ```
2. **Upload the Dataset**: Place the MRI dataset in your Google Colab environment.
3. **Run the Notebook**: Open the notebook in Google Colab, and execute the cells step-by-step.
4. **Visualize Results**: The notebook will output classification results along with visualizations such as confusion matrices and ROC curves.


### Transfer the Folder and Files to drive

Add the data folder,vit.py and utils.py to your Google Drive.

### Mount Google Drive on Google Colab

To mount your Google Drive on Google Colab, follow these steps:

1. Run the following code block in Google Colab:

    ```bash
    from google.colab import drive
    drive.mount('/content/drive')
    ```
2. Follow the link provided and authenticate with your Google account.
3. Copy the authentication code and paste it into the designated area only if you have 2 factor authentication.

### Navigate to Content Directory

Navigate to the content directory in Google Colab:
```bash
%cd /content
```

### Paste vit.py and utils.py
Upload the vit.py and utils.py in Google Drive first.

Upload vit.py and utils.py files to Google Colab. Then, move them to the content directory using the file manager or the following command:

```bash
!mv /content/drive/your_path/vit.py /content
!mv /content/drive/your_path/utils.py /content
```
Replace your_path with the actual path to the files in your Google Drive.

or 

directly drag and move the files from local storage to the content folder

### Run Code Blocks

Now, you are all set to run the code blocks in code.ipynb one by one. Follow the instructions within the notebook for each code block.

### Example Output:
- Tumor detected: **Yes**
- Tumor type: **Glioma**

## Results and Performance
- Achieved  **higher accuracy** on the test set.
- Precision, recall, and F1-scores for each tumor class.
- Confusion matrix showing model performance on different tumor types.

## Future Improvements
- **Hyperparameter Tuning**: Optimize learning rate, batch size, and model architecture for improved performance.
- **Transfer Learning**: Use pre-trained weights from larger datasets to boost model accuracy.
- **Additional Data Augmentation**: Include more aggressive data augmentation techniques to make the model more robust to variations in MRI scans.

## Conclusion
This project demonstrates the potential of Vision Transformers in the medical field, specifically for brain tumor detection and classification. With further improvements, this system could aid doctors in diagnosing brain tumors more efficiently and accurately.

## Acknowledgments
This project is based on publicly available research and datasets. Special thanks to the medical image dataset providers and the open-source community for contributing the Vision Transformer models.
```

