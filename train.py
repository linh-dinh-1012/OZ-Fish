import pandas as pd
import matplotlib.pyplot as plt

from tensorflow import strings
from tensorflow.io import read_file, decode_image, decode_png, decode_jpeg
from tensorflow.data import Dataset, AUTOTUNE
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Rescaling, Resizing, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.dtypes import int32
from tensorflow.image import resize

components = {"FILENAME":0, "FAMILY": 1, "GENUS": 2, "SPECIES": 3}

IMG_HEIGHT=250
IMG_WIDTH=250
NUM_CLASSES=75
BATCH_SIZE=100
PATH ="./raw_data/ozfish-crops/FDFML/crops/"

def get_label(record, component="FAMILY"):
  position = components[component]
  label = record[position]
  return strings.to_number(label, out_type=int32)

def get_image(record):
  image_name = record[0]
  file_path = strings.join([PATH, image_name])
  file = read_file(file_path)
  image = decode_png(file) #decode_png(file) #decode_image(file)
  return resize(image, size=(IMG_HEIGHT, IMG_WIDTH))/255

def process_record(record):
  label = get_label(record, component="FAMILY")
  image = get_image(record)
  return image, label

def get_model():

    model = Sequential([
#      Rescaling(1./255, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
#    Rescaling(1./255),
#      Resizing(width=IMG_WIDTH, height=IMG_HEIGHT, ),
      Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
      MaxPooling2D(),
      Conv2D(32, 3, padding='same', activation='relu'),
      MaxPooling2D(),
      Conv2D(64, 3, padding='same', activation='relu'),
      MaxPooling2D(),
      Flatten(),
      Dense(75, activation='relu'),
      Dense(NUM_CLASSES, activation='softmax')
    ])

    return model

def compile_model(model):
        model.compile(optimizer='adam',
              loss=SparseCategoricalCrossentropy(from_logits=False), # or "categorical_crossentropy"?
              metrics=['accuracy'])

def get_dataset(filename):

    # Read CSV file containing the cropped images details
    df = pd.read_csv(filename, delimiter=",", encoding="latin", header=0)

    X = df.astype(str)
    ds = Dataset.from_tensor_slices(X)
    processed_ds = ds.map(process_record, num_parallel_calls=AUTOTUNE)

    # manage caches
    #dataset = processed_ds.cache().batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)
    dataset = processed_ds.batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)
    return dataset

if __name__ == '__main__':

    # Generate the TF datasets for train and val
    # each dataset element is an image and a label (family)
    # an image has a variable width and height (it is a cropped image from a whole frame)
    train_ds = get_dataset("./train_crop_label.csv")
    val_ds = get_dataset("./val_crop_label.csv") #("./val_crop_sample.csv")

    # TEST: retrieve 1st tf train record
    # for image, label in train_ds.take(1):
    #     #print(image[0].shape)
    #     plt.imshow(resize(image[1], (IMG_HEIGHT, IMG_WIDTH)))
    #     #print(type(image))
    #     #plt.title(label[0])
    # plt.show()

    # Train the model
    model = get_model()
    compile_model(model)
    #model.summary()

    # Checkpoint to save intermediary weights
    checkpoint_filepath = './tmp/checkpoint'
    model_checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)

    # Early Stopper
    early_stopper = EarlyStopping(patience = 10,
        monitor="val_loss",
        restore_best_weights=True)

    # Fit model
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=3,
        callbacks=[model_checkpoint_callback, early_stopper],
        batch_size=BATCH_SIZE
        )

    # Save the entire model as a `.keras` zip archive.
    model.save('crop_model.keras')
