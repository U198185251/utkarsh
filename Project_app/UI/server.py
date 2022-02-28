from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/', methods=['GET'])
def root():
    return render_template('index.html')

@app.route('/click', methods=['post'])
def click():
    # click = int(request.form['click'])
    return render_template('click.html')


app.run(host='0.0.0.0',port='2000', debug=True)