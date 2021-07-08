#import the important libraries 
import argparse
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub 
import numpy as np
import matplotlib.pyplot as plt
import json
import glob
from PIL import Image



def process_image(image):
    
    image = tf.convert_to_tensor(image)
    resized_imag = tf.image.resize(image, (224,224)).numpy()
    normalized_image=resized_imag/255
    return normalized_image



def predict(image_path, model, top_k):
    im = Image.open(image_path)
    test_image = np.asarray(im)
    processed = process_image(test_image)

    pred = model.predict(np.expand_dims(processed, axis = 0))
    pred= pred[0].tolist()
    probs, classes= tf.math.top_k(pred, k=top_k)
    
    probs = probs.numpy().tolist()
    classes_label = classes.numpy().tolist()
    classes=[class_names[str(value+1)] for value in classes_label]

    print("Classes",classes)
    return probs, classes




if __name__ == '__main__':
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path')
    parser.add_argument('loaded_model')
    parser.add_argument('--top_k',type=int,default=5)
    parser.add_argument('--category_names',default='label_map.json')  
    print('predict.py, running')
    args = parser.parse_args()
    
    #print the arguments 
    print(args)
    print('image_path:', args.image_path)
    print('loded_model:', args.loaded_model)
    print('top_k:', args.top_k)
    print('category_names:', args.category_names)
    
    with open(args.category_names, 'r') as f:
        class_names = json.load(f)
    
    
    image_path = (args.image_path)  
    model = tf.keras.models.load_model(args.loaded_model ,custom_objects={'KerasLayer':hub.KerasLayer} )
    top_k = (args.top_k)
    
    #use the predefined function to predict 
    probs, classes = predict(image_path, model, top_k)
    
    print('The predicted flowers name:\n',classes)
    print('The Probabilities: \n ', probs)