from flask import Flask, render_template
from data_processing import load_data

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')
    print(load_data())


if __name__ == '__main__':
    app.run(debug=True)

