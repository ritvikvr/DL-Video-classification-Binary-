# DL-Video-classification-Binary-

CNN Video Classification – Violence Detection
Overview
This project implements a video frame classification pipeline to detect violent vs non-violent content using deep learning and transfer learning (ResNet50). It demonstrates an efficient workflow for leveraging pre-trained convolutional neural networks (CNNs) on image data extracted from video datasets.

Dataset
Source: [Real-Life Violence and Non-Violence Dataset – Kaggle]

Contents: 11,063 frames (images) labeled as violence or non_violence

Split:

70% Training

15% Validation

15% Test

├── 23011101153_CNN_VideoClassification.ipynb        # Main Jupyter Notebook
├── CNN_Video_Classification_Analysis.pdf            # Project analysis report
├── README.md                                        # Project documentation
└── data/
     └── real-life-violence-and-non-violence-data/   # (Downloaded from Kaggle)

Setup & Requirements
Python 3.7+

TensorFlow 2.x

Keras

NumPy, Matplotlib, Pillow

Kaggle API for dataset download

pip install tensorflow kagglehub matplotlib numpy pillow

Data Preparation
Download the dataset from Kaggle.

Extract video frames (images) into separate folders: violence/ and non_violence/.

Images will be auto-resized to 224x224 pixels during preprocessing.

Preprocessing
Resize: All images to 224x224 (for ResNet input)

Label Mode: Automatic from directory names (binary)

Normalization: ResNet50-specific preprocessing

Pipeline: Shuffle, batch, and prefetch for optimized GPU usage

Model Architecture
Base: ResNet50 (ImageNet pre-trained, all layers frozen)

Custom Head:

GlobalAveragePooling2D

Dropout (0.5)

Dense(1, activation='sigmoid') for binary output

Trainable Parameters: Only final classifier layers (approx. 2,049)

Training
Epochs: 5

Batch Size: 32

Learning Rate: 0.001

Optimizer: Adam

Loss Function: Binary Crossentropy

Metric                   |  Value     
-------------------------+------------
Final Training Accuracy  |  92.98%    
Final Validation Acc.    |  95.22%    
Test Accuracy            |  94.90%    
Training Time            |  ~2.5 hours

Key Features
Transfer learning using powerful ResNet50 backbone

Strong spatial feature extraction from video frames

Accelerated training by freezing pre-trained layers

Optimized data pipeline with tf.data.AUTOTUNE and prefetch

High classification performance on challenging video data

Modular and reproducible Jupyter Notebook workflow

Limitations and Future Work
Temporal Features: Current approach only considers individual frames; does not model temporal (motion) dynamics. For full video understanding, consider 3D CNNs, LSTMs, or Transformers for spatio-temporal modeling.

Dataset Specificity: Results are validated on provided dataset; generalization to unseen video domains should be tested.

References
23011101153_CNN_VideoClassification.ipynb (Project Code)

Real-Life Violence and Non-Violence Dataset (Kaggle)

How to Run:

Download and extract dataset as specified.

Run the provided Jupyter Notebook.

Follow inline comments and outputs to train and evaluate your model.

