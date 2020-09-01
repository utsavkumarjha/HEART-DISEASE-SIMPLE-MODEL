import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))


@app.route('/')
def home():
    return render_template('trytwo.html')

@app.route('/predict',methods=['POST','GET'])
def predict():
    input_features = [float(x) for x in request.form.values()]
    features_value = [np.array(input_features)]
    
    features_name =  ["age","sex", "cp", "trestbps", "chol", "thalach", "exang",
                    "oldpeak", "slope", "ca", "thal"]
    df2 = pd.DataFrame(features_value, columns=features_name)
    output = model.predict(df2)
    
    if output == 1:
        res_val = "**Heart Disease**"
    else:
        res_val = "No Heart Disease"
        
    return render_template('trytwo.html', prediction_text='Patient has {}'.format(res_val))
if __name__ == "__main__":
    app.run()