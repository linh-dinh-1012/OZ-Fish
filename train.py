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
from tensorflow.keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split

components = {"FILENAME":0, "FAMILY": 1, "GENUS": 2, "SPECIES": 3}

IMG_HEIGHT=250
IMG_WIDTH=250
#NUM_CLASSES=75
BATCH_SIZE=100
PATH ="../raw_data/ozfish-crops/FDFML/crops/"
CROP_FILE_CSV="./crop_labeled_simplified.csv"

def preprocess_files(filename, threshold_families, threshold_balance, proportion):
    df = pd.read_csv(filename, delimiter=",", encoding="latin")
    df = remove_small_families(df,threshold_families)
    df = cut_big_classes(df,threshold_balance,proportion)

    nb_families = df['family_id'].nunique()
    #nb_families = df['family_id'].max() +1
    print(f"number of families:{nb_families}")
    df_train, df_test = train_test_split(df, stratify=df.loc[:,"family_id"])
    df_train, df_val = train_test_split(df_train, stratify=df_train.loc[:,"family_id"])

    print(f"train:{df_train.shape}")
    print(f"val:{df_val.shape}")
    print(f"test:{df_test.shape}")

    return nb_families, get_dataset_from_df(df_train), get_dataset_from_df(df_val), get_dataset_from_df(df_test)

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

def get_label(record, component="FAMILY"):
  position = components[component]
  label = record[position]
  return strings.to_number(label, out_type=int32)

def get_image(record):
  image_name = record[0]
  file_path = strings.join([PATH, image_name])
  file = read_file(file_path)
  image = decode_png(file)
  return resize(image, size=(IMG_HEIGHT, IMG_WIDTH))/255

def process_record(record):
  label = get_label(record, component="FAMILY")
  image = get_image(record)
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

if __name__ == '__main__':

    # Generate the TF datasets for train and val
    # each dataset element is an image and a label (family)
    # an image has a variable width and height (it is a cropped image from a whole frame)

    # train_ds = get_dataset("./train_crop_label.csv")
    # val_ds = get_dataset("./val_crop_label.csv") #("./val_crop_sample.csv")

    nb_families, train_ds, val_ds, test_ds = preprocess_files(CROP_FILE_CSV, 150, 3000, 0.7)

    # nb_families=75 # why ?

    # TEST: retrieve 1st tf train record
    # for image, label in train_ds.take(1):
    #     #print(image[0].shape)
    #     plt.imshow(resize(image[1], (IMG_HEIGHT, IMG_WIDTH)))
    #     #print(type(image))
    #     #plt.title(label[0])
    # plt.show()

    # Train the model
    model = get_model(nb_families)
    compile_model(model)
    print(model.summary())

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
        epochs=3,
        callbacks=[model_checkpoint_callback, early_stopper, tensorBoard],
        batch_size=BATCH_SIZE
        )

    model.evaluate(test_ds)

    # Save the entire model as a `.keras` zip archive.
    model_folder = "models/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    model.save(model_folder+'crop_model.keras')
