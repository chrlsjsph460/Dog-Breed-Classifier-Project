from flask import Flask, flash, request, redirect, url_for, render_template
import urllib.request
import io
from PIL import Image
import cv2
import numpy as np
from base64 import b64encode


app = Flask(__name__)
 
 
app.secret_key = "a585f83faf56c7bf499b2d6e2a593a8742cc44f5"
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
 
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

from python_backend import my_algo

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
     
@app.route('/home')
def home():
    return render_template('index.html')
 
@app.route('/classify', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    del request
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        # get bytes stream of file
        file_string = file.read()
        # convert string to np array
        np_image = np.fromstring(file_string, np.uint8)
        del file_string
        # convert np_image to image
        img = cv2.imdecode(np_image, cv2.IMREAD_COLOR)
        del np_image
        # change format to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # process the image and get breed message
        breed_msg = my_algo(img)
        
        #get the data url 
        #https://stackoverflow.com/questions/59581565/how-to-display-flask-image-to-html-directly-without-saving
        file_object = io.BytesIO()
        # get the image type. Example... jpeg
        mime = file.mimetype
        del file
        # get the image and save as stream
        img =Image.fromarray(img.astype('uint8'))
        img.save(file_object, mime.split("/")[1])
        del img
        # convert image to base64 
        encode = b64encode(file_object.getvalue()).decode("ascii")
        file_object.close()
        del file_object
        # construct dataURL
        base64img = f'data:{mime};base64,{encode}'
        del mime
        del encode
        
        return render_template('index.html', msg = breed_msg, img = base64img)

    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return redirect(request.url)
 

@app.route('/notebook')
def notebook():
    return render_template("notebook_printout.html", code=301)
 
if __name__ == "__main__":
    app.run(debug=True)


#https://stackoverflow.com/questions/62348356/decode-image-bytes-data-stream-to-jpeg
#https://www.declarecode.com/code-solutions/python/werkzeug-datastructures-filestorage-to-numpy
