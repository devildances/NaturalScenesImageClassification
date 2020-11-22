import pandas as pd, numpy as np
import os, random, datetime
import matplotlib.pyplot as plt, seaborn as sns
from matplotlib.image import imread
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, BatchNormalization, Dropout
from tensorflow.keras.activations import relu, softmax
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.losses import sparse_categorical_crossentropy, categorical_crossentropy
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint
from tensorflow.keras.preprocessing import image
from sklearn.metrics import classification_report, confusion_matrix

class pathDir:
    __train_dir = "../../DATA/NaturalSceneImages/seg_train/seg_train/"
    __test_dir = "../../DATA/NaturalSceneImages/seg_test/seg_test/"
    __pred_dir = "../../DATA/NaturalSceneImages/seg_pred/seg_pred/"
    __pretrained_dir = "../../PRETRAINEDMODEL/"

    def loadDir(self):
        return [pathDir.__train_dir, pathDir.__test_dir, pathDir.__pred_dir]

    def pmDir(self):
        return pathDir.__pretrained_dir

def trainSampleView():
    train_dir = pathDir().loadDir()[0]
    categories = os.listdir(path=train_dir)
    fig, ax = plt.subplots(nrows=5, ncols=len(categories), figsize=(30,25))

    for row in range(5):
        for col, labl in zip(range(len(categories)), categories):
            idx = np.random.randint(low=0, high=len(os.listdir(path=train_dir+labl)))
            ax[row][col].imshow(imread(train_dir+labl+'/'+os.listdir(path=train_dir+labl)[idx]))
            ax[row][col].axis('off')
            ax[row][col].set_title(str(labl))

    plt.show()

def dataInfo(img_res=False):
    train_dir = pathDir().loadDir()[0]
    test_dir = pathDir().loadDir()[1]
    df_tr, df_ts, dim1, dim2, res = {}, {}, [], [], 0

    for labl in os.listdir(train_dir):
        for img_file in os.listdir(train_dir+labl):
            img = imread(train_dir+labl+'/'+img_file)
            dim1.append(img.shape[0])
            dim2.append(img.shape[1])

    for labl in os.listdir(test_dir):
        for img_file in os.listdir(test_dir+labl):
            img = imread(test_dir+labl+'/'+img_file)
            res = img.shape[2]
            dim1.append(img.shape[0])
            dim2.append(img.shape[1])

    img_shape = (int(np.mean(dim1)), int(np.mean(dim2)),res)

    for labl in os.listdir(train_dir):
        df_tr[labl] = len(os.listdir(train_dir+labl))

    for labl in os.listdir(test_dir):
        df_ts[labl] = len(os.listdir(test_dir+labl))

    df_tr = pd.DataFrame(df_tr, index=["total"]).transpose()
    df_tr["part"] = "train"
    df_tr = df_tr.rename_axis('category').reset_index()
    df_ts = pd.DataFrame(df_ts, index=["total"]).transpose()
    df_ts["part"] = "test"
    df_ts = df_ts.rename_axis('category').reset_index()
    df = pd.concat([df_tr,df_ts], axis=0)

    plt.figure(figsize=(8,6))
    sns.barplot(x="category", y="total", palette="YlGnBu", data=df, hue="part")
    plt.ylabel("count")
    plt.xlabel("category")
    plt.title("Total images of each label in train and test datasets")
    plt.show()

    if img_res:
        p = sns.jointplot(dim1, dim2)
        p.fig.suptitle("Average images resolution")
        p.fig.tight_layout()
        p.fig.subplots_adjust(top=0.95)
        plt.show()

    return img_shape

def dataPrep(image_shape=(150,150), color_mode='rgb', batch_size=64, class_mode='categorical', image_augmentation={}):
    train_dir = pathDir().loadDir()[0]
    test_dir = pathDir().loadDir()[1]
    categories = os.listdir(path=train_dir)
    total_categories = len(categories)

    train_generator = ImageDataGenerator(rescale=1/255,
                                        rotation_range=image_augmentation["rotation_range"],
                                        width_shift_range=image_augmentation["width_shift_range"],
                                        height_shift_range=image_augmentation["height_shift_range"],
                                        shear_range=image_augmentation["shear_range"],
                                        brightness_range=image_augmentation["brightness_range"],
                                        zoom_range=image_augmentation["zoom_range"],
                                        channel_shift_range=image_augmentation["channel_shift_range"],
                                        horizontal_flip=image_augmentation["horizontal_flip"],
                                        fill_mode=image_augmentation["fill_mode"])

    test_generator = ImageDataGenerator(rescale=1/255)

    train_data = train_generator.flow_from_directory(directory=train_dir,
                                                    target_size=image_shape,
                                                    color_mode=color_mode,
                                                    batch_size=batch_size,
                                                    class_mode=class_mode,
                                                    shuffle=True)

    test_data = test_generator.flow_from_directory(directory=test_dir,
                                                target_size=image_shape,
                                                color_mode=color_mode,
                                                batch_size=batch_size,
                                                class_mode=class_mode,
                                                shuffle=False)

    labels = {v:k for k,v in train_data.class_indices.items()}

    return categories, total_categories, train_data, test_data, labels

def modeling(input_shape=(150,150,3), output_label=1, verbose=False):

    model = Sequential()
    model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation=relu, input_shape=input_shape))
    model.add(Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), activation=relu))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), activation=relu))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), activation=relu))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(units=512, activation=relu))
    model.add(Dense(units=512, activation=relu))
    model.add(Dropout(rate=0.5))
    model.add(Dense(units=output_label, activation=softmax))

    model.compile(optimizer=Adam(lr=0.0001, decay=1e-6), loss=categorical_crossentropy, metrics=['accuracy'])
    if verbose:
        print(model.summary())
    return model

def trainModel(model, data_train, data_validation, early_stop=False, patience=2, checkpoint=False, model_name=str(int(datetime.datetime.now().timestamp())), num_epochs=10, verbose=0):
    steps_per_epoch = data_train.n // data_train.batch_size
    validation_steps = data_validation.n // data_validation.batch_size
    callbacks = []

    if early_stop:
        es = EarlyStopping(monitor="val_loss", patience=patience, verbose=verbose)
        callbacks.append(es)

    if checkpoint:
        cp = ModelCheckpoint(filepath="models/"+model_name+".h5",
                            monitor="val_loss",
                            save_best_only=True,
                            save_weights_only=False,
                            verbose=verbose,
                            mode='min',
                            save_freq='epoch')
        callbacks.append(cp)

    history = model.fit(data_train,
                        epochs=num_epochs,
                        verbose=verbose,
                        callbacks=callbacks,
                        validation_data=data_validation,
                        steps_per_epoch=steps_per_epoch,
                        validation_steps=validation_steps)

    if checkpoint:
        model = load_model(filepath="models/"+model_name+".h5")

    return model, history

def modelEvaluation(model, history, data_validation):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 6))
    ax[0].plot(epochs, acc, 'r', label='Training accuracy')
    ax[0].plot(epochs, val_acc, 'b', label='Validation accuracy')
    ax[0].legend(loc=0)
    ax[1].plot(epochs, loss, 'r', label='Training loss')
    ax[1].plot(epochs, val_loss, 'b', label='Validation loss')
    ax[1].legend(loc=0)

    plt.suptitle('Train and validation')
    plt.show()

    print("\n\n")
    test_loss, test_acc = model.evaluate(data_validation)
    print("validation accuracy :", str(test_acc*100)+"%")
    print("validation loss :", test_loss)

def modelReport(model, data_validation):
    y_pred = np.argmax(model.predict(data_validation), axis=-1)
    print(classification_report(data_validation.classes, y_pred, target_names=data_validation.class_indices.keys()), end='\n\n\n')

    cm = confusion_matrix(data_validation.classes, y_pred)
    plt.figure(figsize=(12,6))
    sns.heatmap(cm, cmap=plt.cm.Blues, annot=True, fmt='.0f',
                xticklabels=data_validation.class_indices.keys(),
                yticklabels=data_validation.class_indices.keys())
    plt.show()

def predictNewImagesInBatch(model, categories_dict, image_shape):
    pred_dir = pathDir().loadDir()[2]
    fig, ax = plt.subplots(nrows=5, ncols=5, figsize=(25,25))

    for row in range(5):
        for col in range(5):
            idx = np.random.randint(low=0, high=len(os.listdir(path=pred_dir)))
            img_path = pred_dir+os.listdir(path=pred_dir)[idx]
            img = image.load_img(pred_dir+os.listdir(path=pred_dir)[idx], target_size=(image_shape[0], image_shape[1]))
            img = np.expand_dims(image.img_to_array(img), axis=0)
            pred = np.argmax(model.predict(img/255), axis=-1)[0]
            ax[row][col].imshow(imread(img_path))
            ax[row][col].axis('off')
            ax[row][col].set_title(str(categories_dict[pred]))

    plt.show()