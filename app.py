
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from keras.preprocessing import image
import pickle
import numpy as np

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
    # prediction = model.predict([[2, 9, 6]])
      # output = round(prediction[0], 2)
    prediction = model.predict(image_predict)
  
    output = prediction[0][0]
    label = round(prediction[0][1])

    return render_template('index.html', prediction_text='The Image belongs to the classification of {}'.format(output), 
    prediction_label='The Image belongs to the classification of {}'.format(label))


if __name__ == "__main__":
    app.run(debug=True)
