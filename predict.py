from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


(_, _), (x_test, y_test) = mnist.load_data()


x_test = x_test.astype('float32') / 255.0
x_test = x_test.reshape(-1, 28*28) 


y_test_cat = to_categorical(y_test, 10)


model = load_model("mnist_model.h5")

y_pred = model.predict(x_test)
y_pred_class = np.argmax(y_pred, axis=1)


print(classification_report(y_test, y_pred_class))

conf_mat = confusion_matrix(y_test, y_pred_class)
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Prediction")
plt.ylabel("Value")
plt.title("Confusion Matrix")
plt.show()

