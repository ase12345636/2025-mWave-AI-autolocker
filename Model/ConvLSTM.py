from keras import layers
import keras


class ConvLSTM(keras.Model):
    def __init__(self):
        super(ConvLSTM, self).__init__()
        self.convlstm1 = layers.ConvLSTM2D(filters=32, kernel_size=(5, 5), padding="same",
                                           return_sequences=True, activation="relu",)
        self.batchnorm1 = layers.BatchNormalization()
        self.pool1 = layers.MaxPooling3D(pool_size=(1, 2, 2), padding='same',
                                         data_format="channels_last")

        self.convlstm2 = layers.ConvLSTM2D(filters=32, kernel_size=(5, 5), padding="same",
                                           return_sequences=True, activation="relu",)
        self.batchnorm2 = layers.BatchNormalization()
        self.pool2 = layers.MaxPooling3D(pool_size=(1, 2, 2), padding='same',
                                         data_format="channels_last")

        self.convlstm3 = layers.ConvLSTM2D(filters=32, kernel_size=(3, 3), padding="same",
                                           return_sequences=True, activation="relu",)
        self.batchnorm3 = layers.BatchNormalization()
        self.pool3 = layers.MaxPooling3D(pool_size=(1, 2, 2), padding='same',
                                         data_format="channels_last")

        self.flat = layers.Flatten()

        self.dense1 = layers.Dense(64, activation="softmax")

        self.dense2 = layers.Dense(10, activation="softmax")

    def call(self, x):
        x = self.convlstm1(x)
        x = self.batchnorm1(x)
        x = self.pool1(x)

        x = self.convlstm2(x)
        x = self.batchnorm2(x)
        x = self.pool2(x)

        x = self.convlstm3(x)
        x = self.batchnorm3(x)
        x = self.pool3(x)

        x = self.flat(x)

        x = self.dense1(x)

        predict = self.dense2(x)

        return predict

    def build_graph(self, input_shape):
        x = layers.Input(shape=input_shape)
        return keras.Model(inputs=[x], outputs=self.call(x))
