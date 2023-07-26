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
### Problem Description & Overview
This Udacity project applies convolutional neural networks and transfer learning to an image classification problem. This guided project's goal is to classify the images of dogs based on their breed. A function is built that will accept any user-supplied image as input. If a dog is detected in the image, it will provide an estimate of the dog's breed. If a human is detected, it will provide an estimate of the dog breed that the human resembles.The dog breeds dataset comprises 8,351 dog images from 133 dog categories. The dataset is split into 6,680 training dog images, 835 validation dog images, and 836 test dog images. Networks pre-trained on the ImageNet datasets were used as a starting point. The last layer was removed and additional layers were added to tailor the network to dog breed classification problem. These pre-trained networks have already learned features that are useful for most computer vision problems and are much more accurate than building a model from scratch using only the examples provided. A categorical crossentropy loss function was used because this a classification problem with many classes. Metrics used to determine how well the model works are validation accuracy and training accuracy. Early stopping was used so that the model wouldn't end up over- or under-trained. The model with the lowest loss on the validation score was chosen. Training stopped when the validation set's loss did not decrease for ten consectuive epochs. Overall, this was an enjoyable project. Surprisingly, this approach to solving the problem had about an 80% accuracy on the validation set. This was surprising to me because some dog breeds are very similar in appearance. Although I didn’t use generated images in my final project, it was interesting to learn how to do it. 


There are three major components to this project.
1. ETL Pipeline: This part of the project was provided by Udacity.
    - Training, testing, and validation dog files are provided by Udacity. They are loaded using sklearn.dataset.load_file(). Human face files are also provided to evaluate a face detector.
    - The Udacity notebook gives an example of how to used OpenCV's implementation of Haar feature-based cascade classifiers to detect human faces in images. The pre-trained Classifier receives a grayscale image and returns boundary boxes for human faces. From this example a function is built to determine whether human faces are detected.
    - (Q1) The face detector identifies 100% of the 100 human faces presented to it as human. The face detector also falsely identifies 11% of the 100 dog faces presented to it as human.
    -  Because we decided to use OpenCV's [face detector](https://github.com/opencv/opencv/blob/4.x/data/haarcascades/haarcascade_frontalface_alt2.xml), we must communicate to the user that we accept human images only when they provide a clear view of a face. (Q2) This is a reasonable expectation because users volunteer their pictures and there is no harmful consequence to the app being unable to identify a face. If this application was used to identify people entering a building or exiting a building, for example, we would have to improve our model to identify faces even when a clear view is not provided.
    -  A pre-trained ResNet-50 model is used to detect dogs in images. This model receives 4D tensor inputs. Each individual image has shape (1, 224, 224, 3). Getting the 4D tensor ready for ResNet-50, and for any other pre-trained model in Keras, requires some additional processing. First, the RGB image is converted to BGR by reordering the channels. All pre-trained models have the additional normalization step. This is implemented in the imported function [preprocess_input](https://www.tensorflow.org/api_docs/python/tf/keras/applications/resnet50/preprocess_input). The pre-trained model's predict method can be used to classify the image as a dog. Dogs return an index between 151 and 168.

Now that we have a way to format our image for supplying to ResNet-50, we are now ready to use the model to extract the predictions. This is accomplished with the predict method, which returns an array whose  𝑖
 -th entry is the model's predicted probability that the image belongs to the  𝑖
 -th ImageNet category. This is implemented in the ResNet50_predict_labels function below.

By taking the argmax of the predicted probability vector, we obtain an integer corresponding to the model's predicted object class, which we can identify with an object category through the use of this dictionary.
3. ML Pipeline: 
    - Dataset is provided pre-partitioined into training, testing, and validation sets.
    - Using a pre-trained face deterctor from OpenCV, write a function that detects the presence of human faces in an image. 
    - Bottleneck features for [ResNet50](https://keras.io/api/applications/resnet/#resnet50-function), [InceptionV3](https://keras.io/api/applications/inceptionv3/), and [VGG19](https://keras.io/api/applications/vgg/#vgg19-function) were provided by Udacity.
    - Use transfer learning by creating a Sequential model with a few layers and using the Bottleneck features as the input to this model.
    - Best model weights are saved for use in web app. During training, Checkpoints and Earlystopping are used to capture the model with the best validation loss. This model is saved for future use.
4. Flask Web App
    - Allow user to upload an image from local directory. 
    - Classify image as one of the 133 supplied dog breeds.


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

