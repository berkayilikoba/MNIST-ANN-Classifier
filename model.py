from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

def create_model():
    model = Sequential()
    model.add(Dense(units=512, activation="relu", input_shape=(28*28,)))
    model.add(Dense(units=256, activation="relu"))
    model.add(Dense(units=10, activation="softmax"))

    model.compile(optimizer=Adam(), 
                loss="categorical_crossentropy",
                metrics=["accuracy"])
    
    return model