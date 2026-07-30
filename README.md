# cardnet
Group Member Names: Sonia Nath & Berny Perez

Language: python.
ML Subject: image classification.
Task:  correctly classify playing cards.
GOAL: achieve models with high classification accuracy on a dataset of cards.
Models: ResNet-50 and EfficientNetb0
Dataset: images of cards,kaggle:https://www.kaggle.com/datasets/gpiosenka/cards-image-datasetclassification


Description 
	For our final AI project, we will use two popular computer vision architectures Residual net(ResNet) and EfficientNet. The goal is to train two image classification models that will be able to classify cards correctly. Performance measurement will be based on accuracy and precision during the test stage. ResNet and EfficientNet have allowed for advancement in efficient training and resource management in very deep neural networks.  The efficient training on images these architecture offers,  allows for machine learning to automate tasks like image classification and pattern identification in larger image dataset. 
	Residual net (ResNet) is a deep learning architecture designed to enable efficient training of very deep neural networks. It introduces the new technique of finding shortcuts between connections. Shortcut allows the model to learn residual mappings instead of direct transformations.  ResNet architecture helps prevent vanishing gradient problems in very deep models. Also enables building networks with hundred or even thousands layers and allows information to flow directly across layers using skips. For our project that revolves around high classification accuracy, we will use ResNet50.  ResNet-50 is a 50-layer deep convolutional neural network and its architecture consists of 50 deep layers, comprising 16 residual blocks. The ResNet-50 structure offers solid accuracy in classification tasks.
	EfficientNet is a convolution neural network that aims to achieve high performance with fewer computational resources. EfficientNet architecture archives efficient use of computational resources by using a new scaling method that uniformly scales all dimensions of depth,width and resolutions using a compound coefficient. For this project we will use EfficientNet-B0, which is the lightweight model, used for image classification. It requires fewer parameters and less memory to train. (usually deployed on mobile devices).

Method:
Preprocessing: all images will go through a preprocessing algorithm before training. Images will be smooth with gradient filters, image transformation will be used on cards not vertical.  The dataset will be split between 70 training/20 testing/10 validation split. 
Training: Optuna will be used to find the best hyperparameters for the final model.
result: Results will be presented as a table showing the difference between models scores in accuracy,precision,recall, and f1-scores. A confusion matrix will be used to show models' understanding of the data after training.
Future goal or task if time allows
If there is enough time we can use AI methods learned in class to make a better model. Maybe make a decision tree between the models to produce a higher classification accuracy. Maybe combine both models, train a high accuracy ResNet model, then use adaboosting to train smaller lightweight efficientNet models.(this is because ResNet is heavier and expensive to train compared to efficientNet). 
Contribution:
Berny Perez 
ResNet-50 Model
Preprocessing data
Sonia Nath
efficientNet-B0 model
Evaluation and preparing results
both 
Work on documents.

Resources:

ResNet- https://www.ultralytics.com/blog/what-is-resnet-50-and-what-is-its-relevance-in-computer-vision#key-features-of-resnet-50
https://www.geeksforgeeks.org/deep-learning/residual-networks-resnet-deep-learning/

EfficientNet-
https://www.ultralytics.com/blog/what-is-efficientnet-a-quick-overview
https://www.geeksforgeeks.org/computer-vision/efficientnet-architecture/

Image classification-
https://www.ibm.com/think/topics/image-classification
https://www.geeksforgeeks.org/computer-vision/what-is-image-classification/

