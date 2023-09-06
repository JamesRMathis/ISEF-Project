from flask import Flask, send_from_directory, request
import firebase_admin
from firebase_admin import credentials, firestore


app = Flask(__name__)

cred = credentials.Certificate(app.static_folder + '/firebase.json')
firebase_admin.initialize_app(cred)
db = firestore.client()

@app.route('/')
def submit():
    return send_from_directory('static', 'index.html')

@app.route('/submit', methods=['POST'])
def submit_code():
    try:
        code = request.form['code']
        behavior = request.form['behavior']
        name = request.form['name']
        email = request.form['email']

        if behavior == 'halt':
            behavior = 1
        else:
            behavior = 0

        data = {'code': code, 'behavior': behavior, 'name': name, 'email': email}

        doc_ref = db.collection('functions').add(data)

        return send_from_directory('static', 'success.html')
    except Exception as e:
        return send_from_directory('static', 'failure.html')

if __name__ == '__main__':
    app.run()
