import numpy as np
import PIL.Image as Image
import matplotlib.pylab as plt
import tensorflow as tf
import sys, os
#===============================
if len(sys.argv) < 2:
    print("SINTAX ERROR: python3 classify_cat.py <IMAGE_NAME>")
    exit(0)

data_root = '/home/venkopad/faculdade/trab-5-visao/cats'

batch_size = 32
img_height = 224
img_width = 224

train_ds = tf.keras.utils.image_dataset_from_directory(
  str(data_root),
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size
)

#===============================
#List the names of classes encountered
class_names = np.array(train_ds.class_names)

IMAGE_SHAPE = (224, 224)

# Restore the model
export_path = "retrained/saved_models/cat_breads"
model = tf.keras.models.load_model(export_path)

# Image that will be classified
directory = os.getcwd() + "/" #Get current directory
little_cat = directory + "/" + sys.argv[1] #Image that will be classified
little_cat = Image.open(little_cat).resize(IMAGE_SHAPE)

little_cat = np.array(little_cat)/255.0

# Add a batch dimension (with np.newaxis) and pass the image to the model:
result = model.predict(little_cat[np.newaxis, ...])

#The result is a 1001-element vector of logits, rating the probability of each class for the image.
#The top class ID can be found with tf.math.argmax:
predicted_class = tf.math.argmax(result[0], axis=-1)

###Decode the predictions
predicted_labels = class_names[predicted_class]

plt.imshow(little_cat)
plt.axis('off')
predicted_class_name = predicted_labels[predicted_class]
_ = plt.title("Prediction: " + predicted_class_name.title())
plt.show()


