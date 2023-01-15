# Dog-Breed-Classifier-Project (Udacity DS Nanodegree Project Four)

## Table of Contents
1.  [Description](#description)
2.  [Getting Started](#getting-started)
3.  [Authors](#authors)
4.  [Licence](#license)
5.  [Acknowledgements](#acknowledgements)
6.  [Screenshots](#screenshots)

## Description
This Udacity project applies convolutional neural networks and transfer learning to an image classification problem. This guided project's goal is to classify the images of dogs based on their breed. A function is built that will accept any user-supplied image as input. If a dog is detected in the image, it will provide an estimate of the dog's breed. If a human is detected, it will provide an estimate of the dog breed that is most resembling.

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
- Running the ETL Pipeline
    1. Navigate to the project's data directory. 
    2. In the terminal execute `python process_data.py disaster_messages.csv disaster_categories.csv clean_messages.db`
    
- Running the ML Pipeline
    1. Navigate to the project's models directory. 
    2. In the terminal execute `python train_classifier.py ..\data\clean_messages.db saved_model.pkl`
    3. To view model scores, execute `python eval_model.py`.  
    
    ![Model Scores](ModelScores.PNG "Model F1, Precision, and Recall by category")

    *Using the `class_weight = "balanced"` parameter greatly improved model performance. `child_alone` scores are 0.0 because no positive cases were provided. It is  benefical to find examples elsewhere.*
    
- Running the Web App
    1. Navigate to the project's app directory. 
    2. In the terminal execute `python run.py`
    3. Visit [localhost:3001](http://localhost:3001/) to view the web app
        
## Authors
     -Charles Joseph
## License
[MIT OPENSOURCE LICENSE](LICENSE.TXT)
## Acknowledgements
- [Udacity](https://www.udacity.com/) for designing the project, providing images, and templates. 
- [The Keras Blog and Francois Chollet]([https://appen.com/](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html)) for providing a tutorial on image augmentation, transfer learning, and model fine-tuning.

## Screenshots
![Webapp Plots](PlotlyPlots.PNG "Webapp Plots")
*The plots show that there is a large disparity in the number of positive cases. It may be possible to improve model by using resampling methods. Additionally, there are some examples with an unrealistic amount of words. These training examples can probably be deleted or word count can be added as a feature.*

![Webapp Search](RunSearchSmoke.PNG "Webapp Search")

*A simple example: "there is smoke" was flagged as related, direct report, and fire.*
