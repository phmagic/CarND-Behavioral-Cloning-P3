import csv
import cv2
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

data_dirs = ['data', 'data_reverse', 'data_edge', 'data_reg', 'track2_reg', 'track2_edge', 'track2_reverse']

generator_batch_size = 32

samples = []
dir_list = []
for folder in data_dirs:
    with open(folder + '/driving_log.csv') as csvFile:
        reader = csv.reader(csvFile)
        for index, line in enumerate(reader):
            if index == 0:
                continue
            line.append(folder)
            samples.append(line)

train_samples, validation_samples = train_test_split(samples, test_size=0.2)

def load_image(img_path, folder):
    img_file = folder + '/IMG/' + img_path.split('/')[-1]
    img = cv2.imread(img_file)
    # Drive.py reads the sim images with RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for line in batch_samples:
                folder = line[len(line) - 1]
                steering_angle = float(line[3])
                should_add_center = False
                if abs(steering_angle) <= 0.2:
                    if np.random.random() < 0.2:
                        should_add_center = True
                else:
                    should_add_center = True

                if should_add_center:
                    image = load_image(line[0], folder)
                    images.append(image)
                    angles.append(steering_angle)

                    # Add double the images
                    images.append(np.fliplr(image))
                    angles.append(-steering_angle)

                    # Add the left and right cameras
                    # with some steering corrections
                    lr_correction = 0.2
                    left = load_image(line[1], folder)
                    images.append(left)
                    angles.append(steering_angle + lr_correction)

                    right = load_image(line[2], folder)
                    images.append(right)
                    angles.append(steering_angle - lr_correction)

            X_train = np.array(images)
            y_train = np.array(angles)
            # print('X_train: ', X_train.shape)
            yield shuffle(X_train, y_train)

def resize(x):
    # https://discussions.udacity.com/t/keras-lambda-to-resize-seems-causing-the-problem/316247/2
    from keras.backend import tf as ktf
    return ktf.image.resize_images(x, (80, 160))

train_generator = generator(train_samples, batch_size=generator_batch_size)
validation_generator = generator(validation_samples, batch_size=generator_batch_size)

print('Train samples: ', len(train_samples))

input_shape = (160, 320, 3)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, Dropout, Cropping2D
from keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint

model = Sequential()
# Normalization
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=input_shape))
# Crop image
model.add(Cropping2D(cropping=((40, 25), (0, 0))))
# Resize image
model.add(Lambda(resize))
model.add(Conv2D(24, kernel_size=(5,5), strides=(2,2), activation='relu'))
model.add(Conv2D(36, kernel_size=(5,5), strides=(2,2), activation='relu'))
model.add(Conv2D(48, kernel_size=(5,5), strides=(2,2), activation='relu'))
model.add(Conv2D(64, kernel_size=(3,3), strides=(1,1), activation='relu'))
model.add(Conv2D(64, kernel_size=(3,3), strides=(1,1), activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.2))
model.add(Dense(50))
model.add(Dropout(0.2))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

csv_logger = CSVLogger('nvidia_training.log')
early_stop = EarlyStopping(min_delta=0.0001)
model_checkpoint = ModelCheckpoint('nvidia.{epoch:02d}-{val_loss:.2f}.hdf5')

model.fit_generator(train_generator,
                    steps_per_epoch=len(train_samples)/generator_batch_size,
                    validation_data=validation_generator,
                    validation_steps=len(validation_samples)/generator_batch_size,
                    epochs=10,
                    callbacks=[csv_logger, early_stop])

model.save('model.h5')

exit()