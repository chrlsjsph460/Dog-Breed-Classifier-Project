import numpy as np
from tensorflow.keras.models import load_model
import cv2  
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image                  
from tqdm import tqdm

# load array with names of dogs from classifier breed[0] = dog_names[0]
dog_names = np.load("dog_names.npy")

def extract_VGG19(tensor):
	from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
	return VGG19(weights='imagenet', include_top=False).predict(preprocess_input(tensor))

def extract_Resnet50(tensor):
	from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
	return ResNet50(weights='imagenet', include_top=False).predict(preprocess_input(tensor))


# extract pre-trained face detector
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')


def face_detector(img):
    """
    returns True if face is detected in image
    Inputs:
        img is cv2 image type     
    Outputs: 
        bool
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0
 

# define ResNet50 model
ResNet50_model = ResNet50(weights='imagenet')

def img_to_tensor(img):
    """
    converts image to 4D tensor
    Input - cv2 image
    """
    # resize image to 224 by 224 keeping the aspect ratio fixed.
    img = image.smart_resize(img, size=(224, 224))
    # convert to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def ResNet50_predict_labels(img):
    """
    returns prediction for img using ResNet50. 
    outputs are labels for imagenet classification task. 
    Indices 151  through 268 inclusive correspond to dogs.
    Input - cv2 image
    
    """

    img = preprocess_input(img_to_tensor(img))
    return np.argmax(ResNet50_model.predict(img))

def dog_detector(img):
    """
    returns "True" if a dog is detected in the image stored at img_path
    Inputs:
        image - cv2 image
    Output:
        boolean
    """
    prediction = ResNet50_predict_labels(img)
    return ((prediction <= 268) & (prediction >= 151)) 

VGG19_model = load_model('saved_models/VGG19_model.hdf5')

def VGG19_predict_breed(img):
    """
    return breed of dog from image
    Input - cv2 image
    Output - string dog breed
    """
    # extract bottleneck features
    bottleneck_feature = extract_VGG19(img_to_tensor(img))
    # obtain predicted vector
    predicted_vector = VGG19_model.predict(bottleneck_feature)
    # return dog breed that is predicted by the model
    return dog_names[np.argmax(predicted_vector)]

### Image classification algorithm
def my_algo(img): 
    """
    returns a string message based on classification of image. 
    identify whether image has human or dog. Identify breed. If unidentifiable, give message. 
    Input - cv2 image
    Output - string message
    """  
    if dog_detector(img):
        breed =  VGG19_predict_breed(img)
        # predict dog breed
        return f"Hello dog! Your predicted dog breed is {breed}"
        
    elif face_detector(img):
        # predict dog breed
        breed =  VGG19_predict_breed(img)
        # predict dog breed
        return f"Hello human! Your predicted dog breed is {breed}"
    
    else:
        return "Hello ??? I don't know what you are."