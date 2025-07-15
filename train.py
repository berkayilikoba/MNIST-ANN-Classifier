from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import to_categorical
from keras.datasets import mnist
import numpy as np 
from model import create_model


(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train = x_train.reshape(-1, 28*28)
x_test = x_test.reshape(-1, 28*28)

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

model = create_model()

early_stopping = EarlyStopping(monitor="val_loss",
                               patience=3,
                               restore_best_weights=True)

checkpoint = ModelCheckpoint("mnist_model.h5", save_best_only=True)

model.fit(x_train, y_train,
          batch_size=32,
          epochs=10,
          validation_split=0.2,
          verbose=1,
          callbacks=[early_stopping, checkpoint])

test_loss, test_acc = model.evaluate(x_test, y_test)

history = model.fit(x_train, y_train,
                    batch_size=32,
                    epochs=10,
                    validation_split=0.2,
                    verbose=1,
                    callbacks=[early_stopping, checkpoint])


import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='Train Accuracy (acc)')
plt.plot(history.history['val_accuracy'], label='Test Accuracy (val_acc)')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

print(f"Test accuracy is: {test_acc:.4f}")


