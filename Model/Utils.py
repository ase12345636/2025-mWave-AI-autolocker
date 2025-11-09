import os
import keras
import matplotlib.pyplot as plt


def plot_model_structure(model, input_shape, out_file_path):
    os.makedirs(os.path.dirname(out_file_path), exist_ok=True)

    model.build((None,) + input_shape)

    keras.utils.plot_model(model.build_graph(input_shape),
                           expand_nested=True,
                           dpi=250,
                           show_shapes=True,
                           to_file=out_file_path)


def plot_learning_curve(history, out_file_path):
    os.makedirs(os.path.dirname(out_file_path), exist_ok=True)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.tight_layout()
    fig = plt.gcf()
    fig.savefig(out_file_path)
    plt.close(fig)
