import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('food_seq_model.h5', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    img_url = request.form.values()
    from keras.models import load_model
    from keras.preprocessing import image
    image_predict = image.load_img(img_url, target_size=(64,64))
    image_predict = image.img_to_array(image_predict)
    image_predict = np.expand_dims(image_predict, axis=0)
    model = load_model('food_seq_model.h5')
    prediction = model.predict(image_predict)
    if result[0][0] == 1:
        output = "Found it!"
    else:
        output = "Image classification is tough. Train me more!!"

    return render_template('index.html', prediction_text='The Image belongs to the classification of {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)
