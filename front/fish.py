import pandas as pd
import os
import numpy as np
import datetime

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import seaborn as sns


from tensorflow import strings, reshape
from tensorflow.io import read_file, decode_image, decode_png, decode_jpeg
from tensorflow.data import Dataset, AUTOTUNE
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Rescaling, Resizing, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.losses import SparseCategoricalCrossentropy, CategoricalCrossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.dtypes import int32
from tensorflow.image import resize
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.utils import img_to_array
from tensorflow.math import confusion_matrix, reduce_sum

from keras.models import load_model
from sklearn.model_selection import train_test_split

# import tensorflow as tf
from ultralytics import YOLO


class Fish:

    # class variables

    # columns
    # nb_families
    # mappings
    # train_ds
    # val_ds
    # test_ds

    # constants
    # IMG_HEIGHT
    # IMG_WIDTH


    def __init__(self, filename, threshold_families, threshold_balance, proportion):

        # init all class constants
        self.IMG_HEIGHT = int(os.environ["IMG_HEIGHT"])
        self.IMG_WIDTH = int(os.environ["IMG_WIDTH"])
        self.EPOCHS = int(os.environ["EPOCHS"])
        self.BATCH_SIZE = int(os.environ["BATCH_SIZE"])
        self.LOCALIZATION_MODEL=str(os.environ["LOCALIZATION_MODEL"])
        self.CLASSIFICATION_MODEL=str(os.environ["CLASSIFICATION_MODEL"])
        self.CROP_PATH=str(os.environ["CROP_PATH"])
        self.BAD_CLASSES=os.environ["BAD_CLASSES"].split(",")

        self.__preprocess(filename, threshold_families, threshold_balance, proportion)

    def train(self):
        #Train the model
        model = self.__get_model()
        self.__compile_model(model)
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
            self.train_ds,
            validation_data=self.val_ds,
            epochs=self.EPOCHS,
            callbacks=[model_checkpoint_callback, early_stopper, tensorBoard],
            batch_size=self.BATCH_SIZE
            )

        print("Displaying graph ...")
        self.__plot_history(history)

        # Save the entire model as a `.keras` zip archive.
        model_folder = "./models/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        os.makedirs(model_folder, exist_ok=True)
        model.save(model_folder+'/crop_model.keras')

        print("Evaluating ...")
        results  = model.evaluate(self.test_ds) # equivalent X_test, y_test
        print("test loss, test acc:", results)
        self.__plot_confusion_matrix(model, self.test_ds)

    def predict(self, image): #image is a string file name

        # Load the YOLO model for Bbox localization
        yolo_model = YOLO(self.LOCALIZATION_MODEL)  # reuse fish trained model
        results = yolo_model(source=image)  # predict on an image

        # Load the KERAS model for Bbox content classification
        fish_classification_model = load_model(self.CLASSIFICATION_MODEL)
        # fish_classification_model.summary()
        img = mpimg.imread(image)
        for result in results:
            boxes = result.boxes.cpu().numpy()
            name =[]
            for i, box in enumerate(boxes):
                r = box.xyxy[0].astype(int)
                crop = img[r[1]:r[3], r[0]:r[2]]

                resized_crop = resize(crop, size=(self.IMG_HEIGHT, self.IMG_WIDTH))/255

                resized_crop = reshape(resized_crop, shape=(1, self.IMG_HEIGHT, self.IMG_WIDTH, 3))
                prediction = fish_classification_model.predict(resized_crop)
                predicted_class_id = np.argmax(prediction)
                class_name = self.__get_name_from_class_id(mapping_df=self.mappings, normalized_class_id=predicted_class_id)
                name.append(class_name)
                print(f"predicted fish:{class_name}")
        return name

    def __cut_selected_classes(self, df):
        df = df[~df["normalized_family_id"].isin([int(c) for c in self.BAD_CLASSES])]
        return df

    def __plot_confusion_matrix(self, model, test_ds):
        true_labels = []
        predictions = []

        # Iterate over the test dataset to collect true labels and predictions
        for images, labels in test_ds:
            preds = model.predict(images, verbose=0)
            preds = np.argmax(preds, axis=1)  # Convert predictions to label index
            true_labels.extend(labels.numpy())
            predictions.extend(preds)

        # Convert lists to numpy arrays
        true_labels = np.array(true_labels)
        predictions = np.array(predictions)

        # Generate the confusion matrix
        cm = confusion_matrix(true_labels, predictions)

        print("Matrix Confusion")
        #print(cm)

        # Convert confusion matrix to percentages
        cm_percentage = cm / reduce_sum(cm, axis=1)[:, np.newaxis] * 100

        # Plotting
        plt.figure(figsize=(20, 20))
        sns.heatmap(cm_percentage, annot=True, fmt=".2f", cmap='Blues', xticklabels=True, yticklabels=True)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.show()

    def __preprocess(self, filename, threshold_families, threshold_balance, proportion):
            df = pd.read_csv(filename, delimiter=",", encoding="latin")

            df = self.__remove_small_families(df,threshold_families)
            df = self.__cut_big_classes(df,threshold_balance,proportion)

            self.nb_families = df['family_id'].nunique()
            # print(f"number of families:{self.nb_families}")


            # generate a df with colmuns family_id as index and count as a column
            df1 = pd.DataFrame(df['family_id'].value_counts())

            # create a new column with the index content
            df1["raw_family_id"]=df1.index
            df1.reset_index(inplace=True)

            df1["normalized_family_id"]=df1.index
            df1.drop(columns=["family_id"], axis=1, inplace=True)
            df1.rename(columns={"raw_family_id":"family_id"}, inplace=True)

            mappings = df.merge(df1, on='family_id', how='outer')

            mappings = self.__cut_selected_classes(mappings)


            # initialize components, will be used during the subsequent process_record function
            columns = pd.DataFrame(mappings.columns)[0].to_dict()
            self.columns = {v: k for k, v in columns.items()}

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

            self.mappings = mappings
            self.train_ds=self.__get_dataset_from_df(df_train)
            self.val_ds=self.__get_dataset_from_df(df_val)
            self.test_ds=self.__get_dataset_from_df(df_test)

    # def __remove_small_families(self, dataframe, treshold):
    #     '''removes the classes that have a number of occurences lower than a specified treshold'''
    #     value_counts = pd.DataFrame(dataframe.family_id.value_counts(sort=True, ascending=False))
    #     value_counts.reset_index(inplace=True)
    #     df = value_counts[value_counts["count"].astype(int)>treshold]
    #     df2 = dataframe.merge(df, how='inner', on='family_id')
    #     df2.drop(columns=["count"], inplace=True)
    #     return df2

    def __remove_small_families(self, dataframe, treshold):
        '''removes the classes that have a number of occurences lower than a specified treshold'''
        value_counts = pd.DataFrame(dataframe.family.value_counts(sort=True, ascending=False))
        value_counts.rename(columns={"family": "count"}, inplace=True)
        value_counts['family']=value_counts.index
        value_counts.drop(value_counts[value_counts['count'] < treshold].index, inplace = True)
        dataframe = dataframe.merge(value_counts, how = 'inner', on = 'family')
        return dataframe
    def __cut_big_classes(self, dataframe, treshold : int, proportion : float):
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
    def __get_dataset_from_df(self,df):
        X = df.astype(str)
        ds = Dataset.from_tensor_slices(X)
        processed_ds = ds.map(self.__process_record, num_parallel_calls=AUTOTUNE)

        # manage caches
        #dataset = processed_ds.cache().batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)
        dataset = processed_ds.batch(self.BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)
        return dataset
    def __get_label(self, record, component="normalized_family_id"):
        position = self.columns[component]
        label = record[position]
        return strings.to_number(label, out_type=int32)
    def __get_image(self, record, component="file_name"):
        position = self.columns[component]
        image_name = record[position]
        file_path = strings.join([self.CROP_PATH, image_name])
        file = read_file(file_path)
        image = decode_png(file)
        return resize(image, size=(self.IMG_HEIGHT, self.IMG_WIDTH))/255
    def __process_record(self, record):
        label = self.__get_label(record, component="normalized_family_id")
        image = self.__get_image(record, component="file_name")
        return image, label
    def __get_model(self):
        model = Sequential([
        Conv2D(16, 3, padding='same', activation='relu', input_shape=(self.IMG_HEIGHT, self.IMG_WIDTH, 3)),
        MaxPooling2D(),
        Conv2D(32, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Conv2D(64, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Flatten(),
        Dense(75, activation='relu'),
        Dense(self.nb_families, activation='softmax')
        ])

        return model
    def __compile_model(self, model):
        model.compile(optimizer='adam',
                loss=SparseCategoricalCrossentropy(from_logits=False),
                metrics=['accuracy'])
    def __plot_history(self, history):
        plt.plot(history.history['loss'])
        plt.title('Train loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.show()
    def __get_name_from_class_id(self, mapping_df, normalized_class_id):
        df = mapping_df[mapping_df.normalized_family_id==normalized_class_id]
        family = df.head(1).family
        return family.iloc[0]
