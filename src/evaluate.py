from keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import argparse
from get_data import get_data
from keras_preprocessing.image import ImageDataGenerator
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def m_evaluate(config_file):
    config = get_data(config_file)
    batch = config['img_augment']['batch_size']
    class_mode = config['img_augment']['class_mode']
    te_set = config['model']['test_path']
    model = load_model('models/trained.h5')

    test_gen = ImageDataGenerator(rescale=1.0 / 255)
    test_set = test_gen.flow_from_directory(te_set,
                                            target_size=(225, 225),
                                            batch_size=batch,
                                            class_mode=class_mode,
                                            shuffle=False)

    target_names = list(test_set.class_indices.keys())

    print("Model output shape:", model.output.shape)
    print("Expected classes:", target_names)

    # Predict on test data
    Y_pred = model.predict(test_set, verbose=1)
    y_pred = np.argmax(Y_pred, axis=1)

    print("Model is predicting:", np.unique(y_pred))

    print("Confusion Matrix")
    cm = confusion_matrix(test_set.classes, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=target_names, yticklabels=target_names)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.savefig('reports/Confusion_Matrix.png')

    print("Classification Report")
    report = classification_report(test_set.classes, y_pred, labels=np.arange(len(target_names)), target_names=target_names, output_dict=True)
    df = pd.DataFrame(report).T
    df['support'] = df['support'].astype(int)
    df.to_csv('reports/classification_report.csv')

    print('Classification Report and Confusion Matrix saved in reports folder')


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', default='params.yaml')
    passed_args = args_parser.parse_args()
    m_evaluate(config_file=passed_args.config)
