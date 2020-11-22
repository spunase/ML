
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from keras.preprocessing import image
import pickle
import numpy as np
import os

app = Flask(__name__)

model = load_model('model.h5')
# model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])  
def predict():
    '''
    For rendering results on HTML GUI
    '''
    img_url = request.form.values()

    image_predict = image.load_img('templates/baklava.jpg', target_size=(64,64))
    image_predict = image.img_to_array(image_predict)
    image_predict = np.expand_dims(image_predict, axis=0)

    y_prob = model.predict(image_predict) 
    y_classes = int(y_prob.argmax(axis=-1))

    PATH = os.getcwd()
    img_folder_path = os.path.join(PATH, "images")

    CATEGORY = os.listdir(img_folder_path)
    print(CATEGORY, y_classes)

    print("The class of the image is ", CATEGORY[y_classes])

    return render_template('index.html', prediction_text='The Image belongs to the classification of {}'.format(output), 
    prediction_label='The Image belongs to the classification of {}'.format(label))


if __name__ == "__main__":
    app.run(debug=True)
