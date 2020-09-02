from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np

app = Flask(__name__)
model=pickle.load(open('model.pkl','rb'))

@app.route('/')
def hello_world():
    return render_template("utsav19.html")

@app.route('/predict',methods=['POST','GET'])
def predict():
    # receive the values send by user in three text boxes thru request object -> requesst.form.values()
    int_features=[float(x) for x in request.form.values()]
    final=[np.array(int_features)]
    print(int_features)
    print(final)
    prediction=model.predict_proba(final)
    output='{0:.{1}f}'.format(prediction[0][1], 2)

    if output>str(0.5):
        return render_template('utsav19.html',pred='Your have heart disease.Take good care of yourself.Your probability of having heart disease is {}'.format(output))
    else:
        return render_template('utsav19.html',pred='Your do not have heart disease. Eat healthy,stay healthy. Your probability of having heart disease is {}'.format(output))

if __name__ == '__main__':
      app.run(debug=False)
