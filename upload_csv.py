from flask import Flask, render_template, request
import pandas as pd

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    file.save('static/data.csv')
    return render_template('index.html')

@app.route('/show')
def show():
    return render_template('index.html', data=pd.read_csv('spam.csv'))

if __name__ == '__main__':
    app.run(debug=True)
