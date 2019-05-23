from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)


@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=['GET'])
def predict():
    # assign API variables to an object
    args = request.args.to_dict()
    # create a variable for each parameter in our API request
    eff1 = str(args['e1'])

    eff1 = word_tokenize(eff1)
    # setup or X variable as a list of all the API values from above
    X = np.asarray([eff1])    

    vex_filename = ('./model/type_vertorizer')

    vectorizer = pickle.load(open(vex_filename, 'rb'))

    vectorizer.transform(X)

    # load model that we created
    filename = ('./model/type_model')
    model = pickle.load(open(filename, 'rb'))

    # get our prediction based on our received data
    prediction = model.predict(X.reshape(1,-1))

    # create label encoder and assign it the values from our
    # previously fitted data
    le = LabelEncoder()
    le.classes_ = np.load('./model/types.npy')

    return render_template("result.html", result=le.inverse_transform(prediction))

if __name__ == '__main__':
    app.run()