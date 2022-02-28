from keras.losses import categorical_crossentropy
from keras.models import Sequential
from keras.layers import Conv2D, BatchNormalization
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator


model = Sequential([Dense(32, input_shape=(784, )), ])
# 0 - 1
# RGB = 255, 0, 0 => 1, 0, 0
x_train_generator = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
x_test_generator = ImageDataGenerator(rescale=1. / 255)
x_validation_generator = ImageDataGenerator(rescale=1. / 255)

# generate train and test data
x_train = x_train_generator.flow_from_directory('/home/chitransh/Documents/Project_app/images/DATA/train', target_size=(48, 48), batch_size=64,
class_mode='categorical', color_mode='grayscale')
x_test = x_test_generator.flow_from_directory('/home/chitransh/Documents/Project_app/images/DATA/test', target_size=(48, 48), batch_size=64,
class_mode='categorical', color_mode='grayscale')
x_validation = x_validation_generator.flow_from_directory(
'/home/chitransh/Documents/Project_app/images/DATA/val', target_size=(48, 48), batch_size=64, class_mode='categorical', color_mode='grayscale')

num_features = 64
num_labels = 7
batch_size = 64
epochs = 30
width, height = 48, 48


def createAndSaveModel():
    model = Sequential()
    # 1st convolution layer
    model.add(Conv2D(32, kernel_size=(3, 3), activation='elu', input_shape=(height, width, 1), padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=(3, 3), activation='elu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.5))

    # 2nd convolution layer
    model.add(Conv2D(64, (3, 3), activation='elu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), activation='elu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.5))

    # 3rd convolution layer
    model.add(Conv2D(128, (3, 3), activation='elu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), activation='elu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Flatten())

    # 1st fully connected neural networks
    model.add(Dense(512, activation='elu'))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='elu'))
    model.add(Dropout(0.2))

    model.add(Dense(num_labels, activation='softmax'))

    # model.summary()

    # Compliling the model
    model.compile(loss=categorical_crossentropy,
                  optimizer=Adam(),
                  metrics=['accuracy'])

    # fit the images to the model
    history = model.fit_generator(x_train, steps_per_epoch=4000, epochs=10, validation_data=x_validation, validation_steps=2000)
    # Get training and test loss histories
    training_loss = history.history['loss']
    test_loss = history.history['val_loss']

    # Create count of the number of epochs
    epoch_count = range(1, len(training_loss) + 1)

    # Visualize loss history
    # plt.plot(epoch_count, training_loss, 'r--')
    # plt.plot(epoch_count, test_loss, 'b-')
    # plt.legend(['Training Loss', 'Test Loss'])
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.show()
    # save the model
    json = model.to_json()
    file = open('my_model1.json', 'w')
    file.write(json)
    file.close()

    # save the weights
    model.save_weights('weights1.h5', True)


createAndSaveModel()
#
# def classify(testImageFile):
#     from keras.models import model_from_json
#     filename = testImageFile
#     # read the json model
#     file = open('my_model_1.json', 'r')
#     data = file.read()
#     print(data)
#
#     file.close()
#
#     # classifier will load the model from the data
#     # data -> contents of the my_model.json file
#     classifier = model_from_json(data)
#
#     # load waits
#     classifier.load_weights('weights_1.h5')
#
#     # load the test image
#     from keras.preprocessing import image
#
#     test_image = image.load_img(testImageFile, target_size=(48, 48))
#     test_image = image.img_to_array(test_image)
#     test_image = np.expand_dims(test_image, axis=0)
#     test_image /= 255
#
#     classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#
#     result = classifier.predict(test_image)
#     max_index = np.argmax(result[0])
#     emotions = ('angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise')
#
#     prediction = emotions[max_index]
#
#     print(f'prediction: {prediction}')
#
#
#     # visualize
#     image = cv.imread(filename)
#     cv.imshow('result', image)
#     cv.waitKey(0)
#
#
# # classify('images/single_prediction/cat_or_dog_1.jpg')
# classify('/home/piyush/Documents/PROJECT_DATA/images/DATA/test/Surprise/img947.jpg')
