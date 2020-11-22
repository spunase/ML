import numpy as np
from flask import Flask, request, jsonify, render_template
from keras.models import load_model
from keras.preprocessing import image
import pickle

app = Flask(__name__)

# model = load_model('food_seq_model.h5')
model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    img_url = request.form.values()

    image_predict = image.load_img('baklava.jpg', target_size=(64,64))
    image_predict = image.img_to_array(image_predict)
    image_predict = np.expand_dims(image_predict, axis=0)
    
    prediction = model.predict([[2, 9, 6]])
    output = round(prediction[0], 2)
    # if prediction[0][0] == 1:
    #     output = "Found it!"
    # else:
    #     output = "Image classification is tough. Train me more!!"

    return render_template('index.html', prediction_text='The Image belongs to the classification of {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)
