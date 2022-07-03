from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_marshmallow import Marshmallow
import os
from tensorflow.python.keras.models import load_model
import numpy as np
import json
import bcrypt
from werkzeug.utils import secure_filename
from flask_cors import CORS
import tensorflow as tf
from keras.models import model_from_json

APP_ROOT = os.path.abspath(os.path.dirname(__file__))
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

json_file = open('plantex_json.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("plantex_weight.h5")
print("Loaded model from disk")


# Loading labels
with open('./labels.json', 'r') as f:
    category_names = json.load(f)
    img_classes = list(category_names.values())


# Pre-processing images
def config_image_file(_image_path):
    predict = tf.keras.preprocessing.image.load_img(_image_path, target_size=(224, 224))
    predict_modified = tf.keras.preprocessing.image.img_to_array(predict)
    predict_modified = predict_modified / 255
    predict_modified = np.expand_dims(predict_modified, axis=0)
    return predict_modified


    ####################
def fileUpload():
    imgtarget = os.path.join(app.config['UPLOAD_FOLDER'], 'test')
    if not os.path.isdir(target):
        os.mkdir(target)
    logger.info("welcome to upload`")
    file = request.files['file']
    filename = secure_filename(file.filename)
    destination = "/".join([target, filename])
    file.save(destination)
    session['uploadFilePath'] = destination
    response = "Whatever you wish too return"
    return response
    ###################






# Predicting
def predict_image(image):
    result = loaded_model.predict(image)

    return np.array(result[1])


# Working as the toString method
def output_prediction(filename):
    _image_path = f"images/{filename}"
    img_file = config_image_file(_image_path)
    results = predict_image(img_file)
    probability = np.max(results)
    index_max = np.argmax(results)

    return {
         "prediction": str(img_classes[index_max]),
         "probability": str(probability)
        }


# Init app
app = Flask(__name__)
CORS(app)

# Database
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:didine1302@localhost/plant_safe'
app.config["IMAGE_UPLOADS"] = "./images"

# Init db
db = SQLAlchemy(app)
# Init ma
ma = Marshmallow(app)


# Model class User
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(100), unique=True)
    fullname = db.Column(db.String(100))
    password = db.Column(db.String(100))




    def __init__(self, email, fullname, password):
        self.email = email
        self.fullname = fullname
        self.password = password


# User Schema
class UserSchema(ma.Schema):
    class Meta:
        fields = ('id', 'email', 'fullname', 'password')


# Init schema
user_schema = UserSchema()
users_schema = UserSchema(many=True)



# Model class User

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    confidence = db.Column(db.FLOAT,nullable=False)
    condit = db.Column(db.String(250),nullable=False)
    img = db.Column(db.String(255), nullable=False)
    

    
    def __init__(self,confidence, condit,img):
        self.confidence = confidence
        self.condit = condit
        self.img = img


class PredictionSchema(ma.Schema):
    class Meta:

        fields = ('id', 'confidence', 'condit','img')


# Init schema
pred_schema = PredictionSchema()
pred_schema = PredictionSchema(many=True)
# Create a user
@app.route('/api/users', methods=['POST'])
def add_user():
    email = request.json['email']
    fullname = request.json['fullname']
    password = request.json['password'].encode('utf-8')
    hash_password = bcrypt.hashpw(password, bcrypt.gensalt())

    new_user = User(email, fullname, hash_password)
    db.session.add(new_user)
    db.session.commit()

    return user_schema.jsonify(new_user)


# Login user
@app.route('/api/users/login', methods=['POST'])
def login_user():
    email = request.json['email']
    password = request.json['password'].encode('utf-8')

    user = db.session.query(User).filter_by(email=email)
    _user = users_schema.dump(user)

    if len(_user) > 0:
        hashed_password = _user[0]['password'].encode('utf-8')
        if bcrypt.checkpw(password, hashed_password):
            return users_schema.jsonify(user)

    return jsonify({"message": "Invalid credentials"})


# Get All users
@app.route('/api/users', methods=['GET'])
def get_users():
    all_users = User.query.all()
    result = users_schema.dump(all_users)

    return jsonify(result)
    


# Image prediction
@app.route('/api/predict', methods=['POST'])
def get_disease_prediction():
    target = os.path.join(APP_ROOT, 'images/')

    if not os.path.isdir(target):
        os.mkdir(target)

    file = request.files.get('file')
    filename = file.filename
   
    print (filename)
    destination = '/'.join([target, filename])
    file.save(destination)
    result = output_prediction(filename)
    print(result['probability'])
    print(result['prediction'])
    profile_entry = Prediction(confidence=result['probability'],condit=result['prediction'],img=filename)
    db.session.add(profile_entry)
    db.session.commit()
    ############
   
    
 
    return jsonify(result)


# Run Server
if __name__ == '__main__':
    app.run(debug=True)  
