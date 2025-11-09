from keras import layers
import keras

import einops


class Conv2Plus1D(keras.layers.Layer):
    def __init__(self, filters, kernel_size, padding):
        super().__init__()
        self.seq = keras.Sequential([
            # Spatial decomposition
            layers.Conv3D(filters=filters,
                          kernel_size=(1, kernel_size[1], kernel_size[2]),
                          padding=padding),
            # Temporal decomposition
            layers.Conv3D(filters=filters,
                          kernel_size=(kernel_size[0], 1, 1),
                          padding=padding)
        ])

    def call(self, x):
        return self.seq(x)


class ResidualMain(keras.layers.Layer):
    """
    Residual block of the model with convolution, layer normalization, and the
    activation function, ReLU.
    """

    def __init__(self, filters, kernel_size):
        super().__init__()
        self.seq = keras.Sequential([
            Conv2Plus1D(filters=filters,
                        kernel_size=kernel_size,
                        padding='same'),
            layers.LayerNormalization(),
            layers.ReLU(),
            Conv2Plus1D(filters=filters,
                        kernel_size=kernel_size,
                        padding='same'),
            layers.LayerNormalization()
        ])

    def call(self, x):
        return self.seq(x)


class Project(keras.layers.Layer):
    """
      Project certain dimensions of the tensor as the data is passed through different
      sized filters and downsampled.
    """

    def __init__(self, units):
        super().__init__()
        self.seq = keras.Sequential([
            layers.Conv3D(filters=units,
                          kernel_size=(1, 1, 1),
                          padding='same'),
            layers.LayerNormalization()
        ])

    def call(self, x):
        return self.seq(x)


class ResidualBlock(keras.layers.Layer):
    def __init__(self, filters, kernel_size):
        super().__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.main = ResidualMain(filters, kernel_size)
        self.project = None

    def build(self, input_shape):
        in_channels = int(input_shape[-1])
        if in_channels != self.filters:
            self.project = Project(self.filters)
        super().build(input_shape)

    def call(self, inputs):
        out = self.main(inputs)
        res = inputs
        if self.project is not None:
            res = self.project(res)
        return layers.add([res, out])


class ResizeVideo(keras.layers.Layer):
    def __init__(self, height, width):
        super().__init__()
        self.height = height
        self.width = width
        self.resizing_layer = layers.Resizing(self.height, self.width)

    def call(self, video):
        """
          Use the einops library to resize the tensor.

          Args:
            video: Tensor representation of the video, in the form of a set of frames.

          Return:
            A downsampled size of the video according to the new height and width it should be resized to.
        """
        # b stands for batch size, t stands for time, h stands for height,
        # w stands for width, and c stands for the number of channels.
        old_shape = einops.parse_shape(video, 'b t h w c')
        images = einops.rearrange(video, 'b t h w c -> (b t) h w c')
        images = self.resizing_layer(images)
        videos = einops.rearrange(
            images, '(b t) h w c -> b t h w c',
            t=old_shape['t'])
        return videos


class R2Plue1DConv(keras.Model):
    def __init__(self):
        super().__init__()
        # initial stem
        self.conv = Conv2Plus1D(filters=16, kernel_size=(2, 5, 5), padding='same')
        self.bn = layers.BatchNormalization()
        self.relu = layers.ReLU()
        self.resize1 = ResizeVideo(32 // 2, 32 // 2)
        self.drop1 = layers.Dropout(0.3)

        # residual stages
        self.resblock1 = ResidualBlock(32, (2, 3, 3))
        self.resize2 = ResizeVideo(32 // 4, 32 // 4)
        self.drop2 = layers.Dropout(0.3)

        self.resblock2 = ResidualBlock(32, (2, 3, 3))
        self.resize3 = ResizeVideo(32 // 8, 32 // 8)
        self.drop3 = layers.Dropout(0.3)

        self.resblock3 = ResidualBlock(64, (3, 3, 3))
        self.gap = layers.GlobalAveragePooling3D()
        self.drop4 = layers.Dropout(0.3)

        self.flatten = layers.Flatten()
        self.classifier = layers.Dense(10, activation="softmax")

    def call(self, x, training=False):
        x = self.conv(x)
        x = self.bn(x, training=training)
        x = self.relu(x)
        x = self.resize1(x)
        x = self.drop1(x, training=training)

        x = self.resblock1(x)
        x = self.resize2(x)
        x = self.drop2(x, training=training)

        x = self.resblock2(x)
        x = self.resize3(x)
        x = self.drop3(x, training=training)

        x = self.resblock3(x)
        x = self.gap(x)
        x = self.drop4(x, training=training)

        x = self.flatten(x)
        x = self.classifier(x)

        return x

    def build_graph(self, input_shape):
        x = layers.Input(shape=input_shape)
        return keras.Model(inputs=[x], outputs=self.call(x))
