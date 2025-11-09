import argparse
import numpy as np

from Model.CNN import CNN
from Model.ConvLSTM import ConvLSTM
from Model.R2Plue1DConv import R2Plue1DConv
from Model.Utils import plot_learning_curve, plot_model_structure


def main(args):
    model = args[0]()
    input_shape = args[1]
    output_file_path_model_structure = args[2]
    output_file_path_learning_curve = args[3]

    model.build((None,) + input_shape)
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer="adam",
                  metrics=["accuracy"])

    model.summary()
    plot_model_structure(model, input_shape, output_file_path_model_structure)

    X = np.random.rand(100, *input_shape).astype(np.float32)
    y = np.random.randint(0, 10, size=(100,)).astype(np.int32)

    history = model.fit(X, y, batch_size=64, epochs=30, validation_split=0.2)

    plot_learning_curve(history, output_file_path_learning_curve)


if __name__ == "__main__":
    model_mapping = {"CNN": [CNN, (32, 32, 9), "ModleStructure//CNN_Structure.png", "LearningCurve//CNN_LearningCurve.png"],
                     "ConvLSTM": [ConvLSTM, (3, 32, 32, 3), "ModleStructure//ConvLSTM_Structure.png", "LearningCurve//ConvLSTM_LearningCurve.png"],
                     "R2Plue1DConv": [R2Plue1DConv, (3, 32, 32, 3), "ModleStructure//R2Plue1DConv_Structure.png", "LearningCurve//R2Plue1DConv_LearningCurve.png"]}

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)

    args = parser.parse_args()
    args = model_mapping[args.model]

    main(args)
