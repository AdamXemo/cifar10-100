from keras.datasets import cifar10, cifar100
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Activation, Flatten, Dropout
import matplotlib.pyplot as plt
import numpy  as np

# Dataset
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

# Converting 2d array  with labels to 1d
y_train = y_train.reshape(-1,)
y_test = y_test.reshape(-1,)

# Normalizing data
x_train = x_train/255.0
x_test = x_test/255.0

# Labels
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Our model
model = Sequential()

# Model layers
model.add(Conv2D(64, (3,3), input_shape =(32,32,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dropout(.3))
model.add(Dense(256))
model.add(Dropout(.3))
model.add(Dense(128))
model.add(Dropout(.3))
model.add(Dense(128))



model.add(Dense(100, activation='softmax'))
# Summary
model.summary()

# Compiling
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=["accuracy"])

# Training
model.fit(x_train, y_train, epochs=10, batch_size=1, validation_data=(x_test, y_test))

# Testing
test_loss, test_acc = model.evaluate(x_test, y_test)
# Testing Accuracy
print(f'\nTest accuracy: {round((test_acc*100), 1)}%\n')

model.save('small_image_classification/model.h5')

# Getting model Predictions
predictions = model.predict(x_test)
print(predictions.shape)

# Figure size
plt.figure(figsize=(10,10))

right = 0

wrong = 0

for i in range(len(x_test)):
    if np.argmax(predictions[i]) != y_test[i]:
        wrong += 1
    else:
        right += 1

print(f'Our model predicted right {right} times and wrong {wrong} times.')

# Showing images
for i in range(len(x_test)):
    plt.grid(False)
    plt.imshow(x_test[i], cmap=plt.cm.binary)
    # Label name
    plt.xlabel(f'What is on image: {classes[y_test[i]]}')
    # NN predict with label name
    plt.title(f'Neural Network image predict: {classes[np.argmax(predictions[i])]}')
    plt.show()
        