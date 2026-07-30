# Card Classification: ResNet-50 vs. EfficientNet-B0

Final project for CS 441/541: Artificial Intelligence (Summer 2026)

**Group members:** Sonia Nath & Berny Perez

## Overview

| | |
|---|---|
| **Language** | Python |
| **ML Subject** | Image classification |
| **Task** | Correctly classify playing cards from images |
| **Goal** | Achieve high classification accuracy on a playing card image dataset |
| **Models** | ResNet-50, EfficientNet-B0 |
| **Dataset** | [Cards Image Dataset (Kaggle)](https://www.kaggle.com/datasets/gpiosenka/cards-image-datasetclassification) |

## Description

For our final AI project, we use two popular computer vision architectures — Residual Network (ResNet) and EfficientNet — to train two image classification models capable of correctly identifying playing cards. Model performance is measured primarily via accuracy and precision on a held-out test set.

Both ResNet and EfficientNet represent key advances in training very deep neural networks efficiently, enabling machine learning systems to automate tasks like image classification and pattern recognition on large image datasets.

### ResNet-50

Residual Network (ResNet) is a deep learning architecture designed to enable efficient training of very deep neural networks. It introduces *shortcut connections* ("skip connections"), which allow the model to learn residual mappings rather than direct transformations. This design helps prevent the vanishing gradient problem in very deep models and makes it possible to build networks with hundreds — or even thousands — of layers, since information can flow directly across layers via the skip connections.

For this project, we use **ResNet-50**, a 50-layer deep convolutional neural network built from 16 residual blocks, known for delivering solid accuracy on classification tasks.

### EfficientNet-B0

EfficientNet is a convolutional neural network designed to achieve high performance with fewer computational resources. It achieves this efficiency through a *compound scaling* method that uniformly scales network depth, width, and resolution using a single scaling coefficient.

For this project, we use **EfficientNet-B0**, the lightweight base model in the EfficientNet family. It requires fewer parameters and less memory to train, and is commonly used in resource-constrained settings such as mobile deployment.

## Method

- **Preprocessing:** All images are preprocessed before training — smoothing via gradient filters, and image transformations to correct card orientation (e.g. non-vertical cards). The dataset is split **70% training / 20% testing / 10% validation**.
- **Training:** [Optuna](https://optuna.org/) is used to search for the best hyperparameters for each final model.
- **Results:** Model performance is reported in a comparison table across accuracy, precision, recall, and F1-score. A confusion matrix is also generated for each model to visualize classification behavior across card classes.

## Future Work (time permitting)

- Explore combining the two models — e.g. training a high-accuracy ResNet model, then using AdaBoost to train smaller, lightweight EfficientNet models (since ResNet is heavier and more expensive to train than EfficientNet).
- Explore a decision-tree-based ensembling approach between the two models to improve overall classification accuracy.

## Contributions

| Member | Contributions |
|---|---|
| **Berny Perez** | ResNet-50 model, data preprocessing |
| **Sonia Nath** | EfficientNet-B0 model, evaluation & results |
| **Both** | Documentation |

## Resources

**ResNet**
- [What is ResNet-50 and what is its relevance in computer vision? — Ultralytics](https://www.ultralytics.com/blog/what-is-resnet-50-and-what-is-its-relevance-in-computer-vision#key-features-of-resnet-50)
- [Residual Networks (ResNet) — Deep Learning — GeeksforGeeks](https://www.geeksforgeeks.org/deep-learning/residual-networks-resnet-deep-learning/)

**EfficientNet**
- [What is EfficientNet? A Quick Overview — Ultralytics](https://www.ultralytics.com/blog/what-is-efficientnet-a-quick-overview)
- [EfficientNet Architecture — GeeksforGeeks](https://www.geeksforgeeks.org/computer-vision/efficientnet-architecture/)

**Image Classification**
- [Image Classification — IBM](https://www.ibm.com/think/topics/image-classification)
- [What is Image Classification? — GeeksforGeeks](https://www.geeksforgeeks.org/computer-vision/what-is-image-classification/)
