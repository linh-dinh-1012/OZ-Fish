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
import datetime
from keras.models import load_model
from tensorflow.keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.utils import img_to_array
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from ultralytics import YOLO
from keras.models import load_model
from tensorflow import reshape


#components = {"file_name":0, "family_id": 1, "count": 2, "normalized_family_id": 3}

IMG_HEIGHT=250
IMG_WIDTH=250
EPOCHS=1
BATCH_SIZE=100
PATH ="../raw_data/ozfish-crops/FDFML/crops/"
CROP_FILE_CSV="./crop_labeled.csv"

# components is a dictionnary containing the list of fields in the CSV file, plus some additional ones added as we process
components = {}

def preprocess_files(filename, threshold_families, threshold_balance, proportion):
    global components # so that components is updated and no a local variable

    df = pd.read_csv(filename, delimiter=",", encoding="latin")

    # file_name,family,genus,species,family_genus_species,family_id,genus_id,species_id,species_long
    # A000001_L.avi.5107.806.371.922.448.png,Scaridae,Chlorurus,capistratoides,Scaridae Chlorurus capistratoides,0,0,0,Chlorurus capistratoides
    # components = pd.DataFrame(df.columns)[0].to_dict()
    # {0: 'file_name', 1: 'family', 2: 'genus', 3: 'species', 4: 'family_genus_species', 5: 'family_id', 6: 'genus_id', 7: 'species_id', 8: 'species_long'}

    # inverse the dictionary
    # components = {v: k for k, v in components.items()}
    # {'file_name': 0, 'family': 1, 'genus': 2, 'species': 3, 'family_genus_species': 4, 'family_id': 5, 'genus_id': 6, 'species_id': 7, 'species_long': 8}

    # add normalized_family_id at the end of the dictionnary
    # components["normalized_family_id"]=len(components)
    # {'file_name': 0, 'family': 1, 'genus': 2, 'species': 3, 'family_genus_species': 4, 'family_id': 5, 'genus_id': 6, 'species_id': 7, 'species_long': 8, 'normalized_family_id': 9}

    df = remove_small_families(df,threshold_families)
    df = cut_big_classes(df,threshold_balance,proportion)
    nb_families = df['family_id'].nunique()
    print(f"number of families:{nb_families}")


    # generate a df with colmuns family_id as index and count as a column
    df1 = pd.DataFrame(df['family_id'].value_counts())

    # create a new column with the index content
    df1["raw_family_id"]=df1.index
    df1.reset_index(inplace=True)

    df1["normalized_family_id"]=df1.index
    df1.drop(columns=["family_id"], axis=1, inplace=True)
    df1.rename(columns={"raw_family_id":"family_id"}, inplace=True)

    mappings = df.merge(df1, on='family_id', how='outer')

    # initialize components, will be used during the subsequent process_record function
    components = pd.DataFrame(mappings.columns)[0].to_dict()
    components = {v: k for k, v in components.items()}

#     Data columns (total 11 columns):
#  #   Column                Non-Null Count  Dtype
# ---  ------                --------------  -----
#  0   file_name             25307 non-null  object
#  1   family                25307 non-null  object
#  2   genus                 25307 non-null  object
#  3   species               25307 non-null  object
#  4   family_genus_species  25307 non-null  object
#  5   family_id             25307 non-null  int64
#  6   genus_id              25307 non-null  int64
#  7   species_id            25307 non-null  int64
#  8   species_long          25307 non-null  object
#  9   count                 25307 non-null  int64
#  10  normalized_family_id  25307 non-null  int64

    df_train, df_test = train_test_split(mappings, stratify=mappings.loc[:,"normalized_family_id"])
    df_train, df_val = train_test_split(df_train, stratify=df_train.loc[:,"normalized_family_id"])

    # print(f"train:{df_train.shape}")
    # print(f"val:{df_val.shape}")
    # print(f"test:{df_test.shape}")

    return nb_families, mappings, get_dataset_from_df(df_train), get_dataset_from_df(df_val), get_dataset_from_df(df_test)

def remove_small_families(dataframe, treshold):
    '''removes the classes that have a number of occurences lower than a specified treshold'''
    value_counts = pd.DataFrame(dataframe.family_id.value_counts(sort=True, ascending=False))
    value_counts.reset_index(inplace=True)
    df = value_counts[value_counts["count"].astype(int)>treshold]
    df2 = dataframe.merge(df, how='inner', on='family_id')
    df2.drop(columns=["count"], inplace=True)
    return df2

def cut_big_classes(dataframe, treshold : int, proportion : float):
    '''classes above a treshold get cut in a proportion'''
    dataframe = dataframe.sample(frac=1)
    value_counts = pd.DataFrame(dataframe.family_id.value_counts(sort=True, ascending=False))
    value_counts.rename(columns={"family_id": "count"}, inplace=True)
    value_counts['family_id']=value_counts.index
    ls = value_counts[ value_counts["count"] > treshold]["family_id"].tolist()
    for l in ls:
        big_family_indices = dataframe.index[dataframe['family_id'] == l]
        dataframe = dataframe.drop(big_family_indices[:int(len(big_family_indices) * proportion)])
    return dataframe

def get_label(record, component="normalized_family_id"):
    position = components[component]
    label = record[position]
    return strings.to_number(label, out_type=int32)

def get_image(record, component="file_name"):
  position = components[component]
  image_name = record[position]
  file_path = strings.join([PATH, image_name])
  file = read_file(file_path)
  image = decode_png(file)
  return resize(image, size=(IMG_HEIGHT, IMG_WIDTH))/255

def process_record(record):
  label = get_label(record, component="normalized_family_id")
  image = get_image(record, component="file_name")
  return image, label

def get_model(nb_families):

    model = Sequential([
      Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
      MaxPooling2D(),
      Conv2D(32, 3, padding='same', activation='relu'),
      MaxPooling2D(),
      Conv2D(64, 3, padding='same', activation='relu'),
      MaxPooling2D(),
      Flatten(),
      Dense(75, activation='relu'),
      Dense(nb_families, activation='softmax')
    ])

    return model

def compile_model(model):
    model.compile(optimizer='adam',
            loss=SparseCategoricalCrossentropy(from_logits=False),
            metrics=['accuracy'])

def get_dataset_from_df(df):
    X = df.astype(str)
    ds = Dataset.from_tensor_slices(X)
    processed_ds = ds.map(process_record, num_parallel_calls=AUTOTUNE)

    # manage caches
    #dataset = processed_ds.cache().batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)
    dataset = processed_ds.batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)
    return dataset

def get_dataset(filename):

    # Read CSV file containing the cropped images details
    df = pd.read_csv(filename, delimiter=",", encoding="latin", header=0)
    return get_dataset_from_df(df)


def plot_history(history):
    plt.plot(history.history['loss'])
    plt.title('Train loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.show()

def get_name_from_class_id(mapping_df, normalized_class_id):
    df = mapping_df[mapping_df.normalized_family_id==normalized_class_id]
    family = df.head(1).family
    return family.iloc[0]

TRAIN = False

if __name__ == '__main__':

    # Generate the TF datasets for train and val
    # each dataset element is an image and a label (family)
    # an image has a variable width and height (it is a cropped image from a whole frame)

    # train_ds = get_dataset("./train_crop_label.csv")
    # val_ds = get_dataset("./val_crop_label.csv") #("./val_crop_sample.csv")

    nb_families, mappings, train_ds, val_ds, test_ds = preprocess_files(CROP_FILE_CSV, 150, 3000, 0.7)

    if (TRAIN):


        # TEST: retrieve 1st tf train record
        # for image, label in train_ds.take(1):
        #     #print(image[0].shape)
        #     plt.imshow(resize(image[1], (IMG_HEIGHT, IMG_WIDTH)))
        #     #print(type(image))
        #     #plt.title(label[0])
        # plt.show()


        print("Training ...")
        #Train the model
        model = get_model(nb_families)
        compile_model(model)
        # print(model.summary())

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

        # Tensor Board
        log_folder = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        tensorBoard = TensorBoard(log_dir=log_folder,
                             histogram_freq=1,
                             write_graph=True,
                             write_images=True,
                             update_freq='epoch',
                             profile_batch=2,
                             embeddings_freq=1)
        # Fit model
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=EPOCHS,
            callbacks=[model_checkpoint_callback, early_stopper, tensorBoard],
            batch_size=BATCH_SIZE
            )

        plot_history(history)

        # Save the entire model as a `.keras` zip archive.
        model_folder = "./models/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        os.makedirs(model_folder, exist_ok=True)
        model.save(model_folder+'/crop_model.keras')

        print("Evaluating ...")
        eval = model.evaluate(test_ds) # equivalent X_test, y_test
        # print(eval)

    else:   # PREDICT

        image = "../B000452_L.MP4.40040.png"

        # Load the YOLO model for Bbox localization
        yolo_model = YOLO("./best.pt")  # reuse fish trained model
        results = yolo_model(source=image)  # predict on an image

        SAVED_KERAS_FILE="./models/20231201-155151/crop_model.keras"

        # Load the KERAS model for Bbox content classification
        fish_classification_model = load_model(SAVED_KERAS_FILE)
        # fish_classification_model.summary()

        img = mpimg.imread(image)
        for result in results:
            boxes = result.boxes.cpu().numpy()
            for i, box in enumerate(boxes):
                r = box.xyxy[0].astype(int)
                crop = img[r[1]:r[3], r[0]:r[2]]

                resized_crop = resize(crop, size=(IMG_HEIGHT, IMG_WIDTH))/255

                resized_crop = reshape(resized_crop, shape=(1, IMG_HEIGHT, IMG_WIDTH, 3))
                prediction = fish_classification_model.predict(resized_crop)
                predicted_class_id = np.argmax(prediction)
                class_name = get_name_from_class_id(mapping_df=mappings, normalized_class_id=predicted_class_id)
                print(f"predicted fish:{class_name}")
