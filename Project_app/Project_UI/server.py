from flask import Flask, render_template

app = Flask(__name__)


@app.route('/', methods=['GET'])
def root():
    return render_template("index.html")

app.run(host='0.0.0.0', port=4500, debug=True)

