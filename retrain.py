import numpy as np
import time

import PIL.Image as Image
import matplotlib.pylab as plt

import tensorflow as tf
import tensorflow_hub as hub

import datetime

# ================================Prepare dataset to retrain================================
data_root = '/home/venkopad/faculdade/trab-5-visao/cats'

batch_size = 32
img_height = 224
img_width = 224

#First, load this data into the model using the image data off disk with 
# tf.keras.utils.image_dataset_from_directory, which will generate a tf.data.Dataset:
train_ds = tf.keras.utils.image_dataset_from_directory(
  str(data_root),
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size
)

val_ds = tf.keras.utils.image_dataset_from_directory(
  str(data_root),
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size
)

#List the names of classes encountered
class_names = np.array(train_ds.class_names)

#Second, because TensorFlow Hub's convention for image models is to expect float 
# inputs in the [0, 1] range, use the tf.keras.layers.Rescaling preprocessing layer to achieve this.
normalization_layer = tf.keras.layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y)) # Where x—images, y—labels.
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y)) # Where x—images, y—labels.

#Third, finish the input pipeline by using buffered prefetching 
# with Dataset.prefetch, so you can yield the data from disk without I/O blocking issues.
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
for image_batch, labels_batch in train_ds:
  print(image_batch.shape)
  print(labels_batch.shape)
  break

# ================================Download and prepare model================================
# Classifier model ho will be retrained
mobilenet_v2 ="https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4"

IMAGE_SHAPE = (224, 224)

#Create the feature extractor by wrapping the pre-trained model as a Keras layer with hub.KerasLayer. Use the trainable=False 
# argument to freeze the variables, so that the training only modifies the new classifier layer:
feature_extractor_layer = hub.KerasLayer(
    mobilenet_v2,
    input_shape=(224, 224, 3),
    trainable=False)

#Attach a classification head: To complete the model, wrap the feature extractor layer in a tf.keras.Sequential 
# model and add a fully-connected layer for classification:
num_classes = len(class_names)

model = tf.keras.Sequential([
  feature_extractor_layer,
  tf.keras.layers.Dense(num_classes)
])

#Shows model appearence
model.summary()

# ================================Train the model================================
#Use Model.compile to configure the training process and add a tf.keras.callbacks.TensorBoard callback to create and store logs:
model.compile(
  optimizer=tf.keras.optimizers.Adam(),
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['acc'])

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir,
    histogram_freq=1) # Enable histogram computation for every epoch.

NUM_EPOCHS = 10

#Start training
history = model.fit(train_ds,
                    validation_data=val_ds,
                    epochs=NUM_EPOCHS,
                    callbacks=tensorboard_callback)

#Plot learning curves
acc = history.history['acc']
val_acc = history.history['val_acc']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()

#Saves the model
export_path = "retrained/saved_models/cat_breads"
model.save(export_path)

# ================================Check the predictions================================
#Obtain the ordered list of class names from the model predictions:
predicted_batch = model.predict(image_batch)
predicted_id = tf.math.argmax(predicted_batch, axis=-1)
predicted_label_batch = class_names[predicted_id]

#Plot the model predictions:

plt.figure(figsize=(10,9))
plt.subplots_adjust(hspace=0.5)

for n in range(30):
  plt.subplot(6,5,n+1)
  plt.imshow(image_batch[n])
  plt.title(predicted_label_batch[n].title())
  plt.axis('off')
_ = plt.suptitle("Model predictions")

plt.show()
