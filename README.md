# Dog-Breed-Classifier-Project (Udacity DS Nanodegree Project Four)

## Table of Contents
1.  [Description](#description)
2.  [Getting Started](#getting-started)
3.  [Authors](#authors)
4.  [Licence](#license)
5.  [Acknowledgements](#acknowledgements)
6.  [Screenshots](#screenshots)

## Description
This Udacity project applies convolutional neural networks and transfer learning to an image classification problem. This guided project's goal is to classify the images of dogs based on their breed. A function is built that will accept any user-supplied image as input. If a dog is detected in the image, it will provide an estimate of the dog's breed. If a human is detected, it will provide an estimate of the dog breed that the human resembles.

Problem Statment: Given the picture of a dog, determine the dog's breed. Given the picture of a human, which dog breed does the human most resemble. If an image of something that is not a dog or a human, tell the user you don't know what the image is. Use convolutional neural networks and transfer learning to determine the breeds. (Expand on this).
Breeds are identified based on the names of breeds given. (Expand on this). Metrics used to determine how well the model works are validation accuracy and training accuracy. This guides how we choose best model






There are three major components to this project.
1. ETL Pipeline: This part of the project was provided by Udacity.
2. ML Pipeline: dog_recog.py, dog_app.ipynb, and ... maybe more???
    - Loads data from the SQLite database
    - Splits the det into training and test sets
    - Use ImageDataGenerator to augment images
    - Use transfer learning with VGG19 as base to build classifier.
    - Fine tune model by setting last block of VGG19 to trainable and slowly optimizing model weights with stochastic gradient descent.
    - Use transfer learning with ResNet as base to build classifier.
    - Fine tune model by setting last block of VGG19 to trainable and slowly optimizing model weights with stochastic gradient descent.  
    - Best model weights are saved for use in web app.
3. Flask Web App
    - Allow user to upload an image from local directory. Visualizes dataset 
    - Classify image using one of the classifiers mentioned.


## Getting Started
### Dependencies

- Python 3
- Sklearn
- Numpy
- Tensorlow,
- CV2
- Web App Library: Flask
- Data Visualization Library: Plotly

### Running the Code
- Using the Webapp
    1. Navigate to the project's directory. 
    2. In the terminal execute `python app.py`
    3. On the same machine, in your browser, navigate to [here](http://127.0.0.1:5000/home) to view the web app
    
    ![Model Scores](ModelScores.PNG "Model F1, Precision, and Recall by category")

    *Using the `class_weight = "balanced"` parameter greatly improved model performance. `child_alone` scores are 0.0 because no positive cases were provided. It is  benefical to find examples elsewhere.*
        
## Authors
     -Charles Joseph
## License
[MIT OPENSOURCE LICENSE](LICENSE.TXT)
## Acknowledgements
- [Udacity](https://www.udacity.com/) for designing the project, providing images, and templates. 
- [The Keras Blog and Francois Chollet](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html) for providing a tutorial on image augmentation, transfer learning, and model fine-tuning.

## Screenshots
![Webapp Plots](PlotlyPlots.PNG "Webapp Plots")
*The plots show that there is a large disparity in the number of positive cases. It may be possible to improve model by using resampling methods. Additionally, there are some examples with an unrealistic amount of words. These training examples can probably be deleted or word count can be added as a feature.*

![Webapp Search](RunSearchSmoke.PNG "Webapp Search")
