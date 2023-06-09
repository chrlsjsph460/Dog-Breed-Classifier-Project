# Dog-Breed-Classifier-Project (Udacity DS Nanodegree Project Four)

## Table of Contents
1.  [Description](#description)
2.  [Getting Started](#getting-started)
3.  [Notebooks](#notebooks)
4.  [Authors](#authors)
5.  [Licence](#license)
6.  [Acknowledgements](#acknowledgements)
7.  [Screenshots](#screenshots)
8.  [Improvements](#improvements)

## Description
This Udacity project applies convolutional neural networks and transfer learning to an image classification problem. This guided project's goal is to classify the images of dogs based on their breed. A function is built that will accept any user-supplied image as input. If a dog is detected in the image, it will provide an estimate of the dog's breed. If a human is detected, it will provide an estimate of the dog breed that the human resembles.The dog breeds dataset comprises 8,351 dog images from 133 dog categories. The dataset is split into 6,680 training dog images, 835 validation dog images, and 836 test dog images. Networks pre-trained on the ImageNet datasets were used as a starting point. The last layer was removed and additional layers were added to tailor the network to dog breed classification problem. These pre-trained networks have already learned features that are useful for most computer vision problems and are much more accurate than building a model from scratch using only the examples provided. A categorical crossentropy loss function was used because this a classification problem with many classes. Metrics used to determine how well the model works are validation accuracy and training accuracy. Early stopping was used so that the model wouldn't end up over- or under-trained. The model with the lowest loss on the validation score was chosen. Training stopped when the validation set's loss did not decrease for ten consectuive epochs. Overall, this was an enjoyable project. Surprisingly, this approach to solving the problem had about an 80% accuracy on the validation set. This was surprising to me because some dog breeds are very similar in appearance. Although I didn’t use generated images in my final project, it was interesting to learn how to do it. 





There are three major components to this project.
1. ETL Pipeline: This part of the project was provided by Udacity.
2. ML Pipeline: 
    - Dataset is provided pre-partitioined into training, testing, and validation sets.
    - Using a pre-trained face deterctor from OpenCV, write a function that detects the presence of human faces in an image. 
    - Bottleneck features for [ResNet50](https://keras.io/api/applications/resnet/#resnet50-function), [InceptionV3](https://keras.io/api/applications/inceptionv3/), and [VGG19](https://keras.io/api/applications/vgg/#vgg19-function) were provided by Udacity.
    - Use transfer learning by creating a Sequential model with a few layers and using the Bottleneck features as the input to this model.
    - Best model weights are saved for use in web app. During training, Checkpoints and Earlystopping are used to capture the model with the best validation loss. This model is saved for future use.
3. Flask Web App
    - Allow user to upload an image from local directory. 
    - Classify image as one of the 133 supplied dog breeds, if pos


## Getting Started
### Dependencies

- Python 3
- Sklearn
- Numpy
- Tensorlow,
- Keras
- CV2
- Web App Library: Flask

### Running the Code
- Using the Webapp
    1. Using your command prompt terminal, navigate to the project's directory. 
    2. In the terminal execute `python app.py`
    3. On the same machine, in your browser, navigate to [here](http://127.0.0.1:5000/home) to view the web app

## Notebooks

[Inception Version of Notebook](dog_app_inception.html)

[Resnet Version of Notebook](dog_app_resnet.html)

There is also a VGG19 version of the notebook, but there wasn't enough space to store bottleneck features for all three models.

## Authors
     -Charles Joseph
## License
[MIT OPENSOURCE LICENSE](LICENSE)
## Acknowledgements
- [Udacity](https://www.udacity.com/) for designing the project, providing images, and templates. 
- [The Keras Blog and Francois Chollet](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html) for providing a tutorial on image augmentation, transfer learning, and model fine-tuning.

## Screenshots
![Webapp Plots](Example1_dogapp.png "Webapp Plots")
![Webapp Plots](Example2_dogapp.png "Webapp Plots")
![Webapp Plots](Example3_dogapp.png "Webapp Plots")
    
## Improvements

- Fine tune model by setting last block of VGG19, InceptionV3, or ResNet50 to trainable and optimize model weights with stochastic gradient descent.

- Use data augmentation to expand dataset.

- Alter project to identify more than one entity in a picture. For example, if a picture has more than one dog, identify the breed of each dog. 

