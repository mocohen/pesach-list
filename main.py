from flask import Flask, render_template, request, redirect, url_for
from flask import send_from_directory
from werkzeug.utils import secure_filename
import os

from google.cloud import vision
import io

import json
import pandas as pd

from PIL import Image
import string
from fuzzywuzzy import process
import numpy as np


app = Flask(__name__)

UPLOAD_FOLDER = '/tmp/uploads'
CROPPED_FOLDER = '/tmp/crops'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['CROPPED_FOLDER'] = CROPPED_FOLDER

# @app.route('/')
# def hello_world():
#    return 'Hello World'

def detect_text(path):
    """Detects text in the file."""

    client = vision.ImageAnnotatorClient()

    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.text_detection(image=image)
    texts = response.text_annotations


    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))
    return texts

def build_list():
    list_dict = {}
    with open('list.json') as fp:
        list_dict = json.load(fp)

    ohel_torah = [ing['_source']['product_name_text'] for ing in list_dict['hits']['hits']]
    df = pd.read_csv('Pesach-List-2022.csv')
    ingredients_df = df[df['Category'] == 'Ingredient']

    full_list = np.append(ingredients_df.Item.values, ohel_torah)
    return full_list

def crop_image(im, left, upper, right, lower):
    # Cropped image of above dimension
    # (It will not change original image)
    im1 = im.crop((left, upper, right, lower))

    # Shows the image in image viewer
    return im1

def process_results(texts, im):
    last = []
    c_l, c_t, c_r, c_b = im.size[0], im.size[1], 0, 0
    processed_texts = []
    bboxes = []
    for text in texts:
        if len(text.description.split()) == 1 and 'ingredients' not in text.description.lower():
            l,t = text.bounding_poly.vertices[0].x, text.bounding_poly.vertices[0].y
            r,b = text.bounding_poly.vertices[2].x, text.bounding_poly.vertices[2].y      
            c_l = min(c_l, l)
            c_t = min(c_t, t)
            c_r = max(c_r, r)
            c_b = max(c_b, b)
            
            last.append(text.description)
            if text.description[-1] in ',:':
                processed_texts.append(' '.join(last))
                bboxes.append((c_l, c_t, c_r, c_b))
                last = []
                c_l, c_t, c_r, c_b = im.size[0], im.size[1], 0, 0
    return processed_texts, bboxes




@app.route('/')
def upload_file():
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(CROPPED_FOLDER, exist_ok=True)
    return render_template('upload.html')
    
@app.route('/uploader', methods = ['GET', 'POST'])
def uploader_file():
   if request.method == 'POST':
      f = request.files['file']
      filename = secure_filename(f.filename)
      f_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
      print('post')
      f.save(f_path)
      print(os.listdir(app.config['UPLOAD_FOLDER']))
      return redirect(url_for('success',filename = filename))

@app.route('/uploads/<filename>')
def display_upload(filename):
    #print('display_image filename: ' + filename)
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

@app.route('/crops/<filename>')
def display_crop(filename):
    #print('display_image filename: ' + filename)
    return send_from_directory(app.config["CROPPED_FOLDER"], filename)


@app.route('/success/<filename>')
def success(filename):
    file_path =  os.path.join(app.config['UPLOAD_FOLDER'], filename)
    texts = detect_text(file_path)
    im = Image.open(file_path)

    texts, bboxes = process_results(texts, im)
    full_list = build_list()

    x = []
    for i, (bbox, text) in enumerate(zip(bboxes, texts)):
        highest = process.extract(text,full_list)
        im_c = crop_image(im, *bbox)
        c_name = f'{i}_'+filename
        im_c.save( os.path.join(app.config['CROPPED_FOLDER'], c_name))
        # display(im_c)
        x.append([c_name, text, *highest[:3]])

    # x = {'date':[u'2012-06-28', u'2012-06-29', u'2012-06-30'], 'users': [405, 368, 119]}
    return render_template('results.html', x=x, filename=filename)

if __name__ == '__main__':
   app.run(debug=True)

   