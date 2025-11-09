from keras import layers
import keras


class CNN(keras.Model):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = layers.Conv2D(filters=64, kernel_size=(7, 7),
                                   activation="relu", padding="same")
        self.pool1 = layers.MaxPooling2D(pool_size=(2, 2), padding="same")
        self.dropout1 = layers.Dropout(0.3)

        self.conv2 = layers.Conv2D(filters=64, kernel_size=(5, 5),
                                   activation="relu", padding="same")
        self.pool2 = layers.MaxPooling2D(pool_size=(2, 2), padding="same")
        self.dropout2 = layers.Dropout(0.3)

        self.conv3 = layers.Conv2D(filters=128, kernel_size=(3, 3),
                                   activation="relu", padding="same")
        self.pool3 = layers.MaxPooling2D(
            pool_size=(2, 2), padding="same")
        self.dropout3 = layers.Dropout(0.3)

        self.flat = layers.Flatten()

        self.dense1 = layers.Dense(128, activation="relu")
        self.dropout5 = layers.Dropout(0.5)

        self.dense2 = layers.Dense(10, activation="softmax")

    def call(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        x = self.conv3(x)
        x = self.pool3(x)
        x = self.dropout3(x)

        x = self.flat(x)

        x = self.dense1(x)
        x = self.dropout5(x)

        predict = self.dense2(x)

        return predict

    def build_graph(self, input_shape):
        x = layers.Input(shape=input_shape)
        return keras.Model(inputs=[x], outputs=self.call(x))
