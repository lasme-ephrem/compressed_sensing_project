import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from matplotlib import pyplot as plt

def format_image_classif(img_file, img_width = 224, img_height = 224):
    """
    Cette fonction lit et formate une image pour qu'elle puisse être envoyée au réseau VGG16
    """
    img = load_img(img_file,target_size=(img_height,img_width))
    img_out = img_to_array(img)
    img_out_model = tf.keras.applications.vgg16.preprocess_input(img_out) # RGB -> BGR
    img_out_model = np.expand_dims(img_out_model, axis=0) #batch dimension
    return img_out_model, img

def unformat_image(img):
    """
    Cette fonction inverse le prétraitement appliqué aux images
    """
    img_out=np.squeeze(img)
    img_out[:, :, 0] += 103.939
    img_out[:, :, 1] += 116.779
    img_out[:, :, 2] += 123.68
    img_out = img_out[:, :, ::-1] #BGR -> RGB
    img_out = np.clip(img_out, 0, 255).astype('uint8')
    return img_out

def adversarial_gradient_step(model, img, step_size, target_class):
    with tf.GradientTape() as tape:
        tape.watch(img)
        loss = model(img)[0, target_class]
    grads = tape.gradient(loss, img)
    grads /= tf.math.reduce_std(grads) + 1e-8
    img += step_size * grads
    return img
