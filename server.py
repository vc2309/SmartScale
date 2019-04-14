from flask import Flask, request
import pandas as pd
import json
import numpy as np
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from PIL import Image
import io
app = Flask(__name__)
food_df = pd.read_csv("food_data.csv",index_col="food_name")
classifier = load_model('best_model.h5')
label_to_class = {}
with open('classes.json','r+') as file:
	label_to_class = json.loads(file.read())
class_dict = {v:k for k,v in label_to_class.items()}

@app.route('/getDetails', methods=["POST"])
def getDetails():
	#TO-DO
	# food = run code to get image class
	image = request.files["image"].read()
	image = Image.open(io.BytesIO(image))
	image = prepare_image(image, target=(96, 96))
	class_ = class_dict.get(np.argmax(classifier.predict(image)))
	weight = float(request.args.get("w"))/100
	details = food_df.loc[class_]*weight
	details["food_name"] = class_
	return details.to_json()

def prepare_image(image, target):
    # if the image mode is not RGB, convert it
    if image.mode != "RGB":
        image = image.convert("RGB")

    # resize the input image and preprocess it
    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image/255

    # return the processed image
    return image

app.run()