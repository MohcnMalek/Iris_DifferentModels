from flask import Flask, render_template, request
import pickle
import numpy as np

def load_models():
    return [
        pickle.load(open('knn_model.pkl', 'rb')),
        pickle.load(open('decision_tree_model.pkl', 'rb')),
        pickle.load(open('random_forest_model.pkl', 'rb')),
        pickle.load(open('svm_model.pkl', 'rb'))
    ]

def predict(model, data):
    return model.predict(data)[0]  # Return just the first prediction

app = Flask(__name__)

@app.route('/')
def man():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])

def home():
    data = np.array([[float(request.form['a']), float(request.form['b']), float(request.form['c']), float(request.form['d'])]])
    algorithm = request.form['algorithm']
    models = load_models()
    model_index = ['Decision Tree', 'K-Nearest Neighbors', 'Random Forest', 'SVM'].index(algorithm)
    prediction = predict(models[model_index], data)
    return render_template('result.html', data=int(prediction))

if __name__ == "__main__":
    app.run(debug=True)