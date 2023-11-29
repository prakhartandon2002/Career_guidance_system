from flask import Flask,render_template,request
import pickle
import numpy as np

y1 = pickle.load(open('y1.pkl','rb'))
X = pickle.load(open('X.pkl','rb'))
from joblib import dump, load
clf = load('z.joblib')

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')


@app.route('/career', methods=['post'])
def recommend():
    
    l=request.form
    x_new=[]
    for key, value in l.items():
        x_new.append(value)
    
    
    new_pred  = clf.predict([x_new])
    data = "Prediction : {}".format(y1[y1['Associated Number']==new_pred[0]]['ROLE'])
    
    return render_template('index.html',data=data)

if __name__=='__main__':
    app.run(debug=True)
    