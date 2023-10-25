import json
import os
import subprocess
import numpy as np
from tqdm import tqdm
from keras.utils import np_utils
from keras.preprocessing import sequence
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
import mlflow
from dl_models import DlModels
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow import keras
import time
import pprint
import argparse
import datetime
import seaborn as sns
import matplotlib.pyplot as plt

from keras.preprocessing.text import Tokenizer


class PhishingDataGen(keras.utils.Sequence):

    def __init__(self, filenames, batch_size):
        self.filenames = filenames
        self.batch_size = batch_size
        accepted_chars = " !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[]\\^_`abcdefghijklmnopqrstuvwxyz{|}~"

        self.tokener = Tokenizer(lower=True, char_level=True, oov_token='-n-')
        self.tokener.word_index = json.loads(open("../dataset/char_index").read())

    def __len__(self):
        return (np.ceil(len(self.filenames) / float(self.batch_size))).astype(np.int)

    def __getitem__(self, idx):
        batch_x = self.filenames[idx * self.batch_size: (idx + 1) * self.batch_size]
        urls = []
        labels = []
        for file in batch_x:

            data = subprocess.check_output("cat {}".format(file), shell=True).decode("utf-8").split("\n")

            while '' in data:
                data.remove("")

            for row in data:
                row = row.strip()
                spt = row.split("\t")

                if '' in spt:
                    spt.remove('')

                url = spt[1]

                if spt[0] == "phishing":
                    label = 0
                elif spt[0] == "legitimate":
                    label = 1
                else:
                    label = -1
                    print("err: {}".format(row))
                    continue

                urls.append(url.lower().replace("http://", "").replace("https://", ""))
                labels.append([label])

        vec = np.asanyarray(self.tokener.texts_to_sequences(urls), dtype=object)
        x = sequence.pad_sequences(vec, maxlen=512)
        y = np.asanyarray(labels)
        #print("files: {} idx: {} - vec: {} - label: {}".format(batch_x, idx, vec.shape, y.shape))
        return x, y

class Plotter:

    def plot_graphs(self, list1, list2, save_to, name1, name2, figure_name):
        val, = plt.plot(list1, label=name1)
        train, = plt.plot(list2, label=name2)

        plt.ylabel(figure_name)
        plt.xlabel("epoch")

        plt.legend(handles=[val, train], loc=2)

        plt.savefig("{0}/{1}.png".format(save_to, figure_name))

        plt.close()

    def plot_confusion_matrix(self, confusion_matrix, save_to):

        sns.set()
        plt.rcParams.update({'font.size': 20})
        plt.figure(figsize=(14.0, 7.0))
        #plt.figure(figsize=(18.0, 9.0))

        row_sums = np.asanyarray(confusion_matrix).sum(axis=1)
        matrix = confusion_matrix / row_sums[:, np.newaxis]
        # matrix = [line.tolist() for line in matrix]
        # g = sns.heatmap(matrix, annot=True, fmt='f', xticklabels=True, yticklabels=True)
        g = sns.heatmap(matrix, xticklabels=True, yticklabels=True, linewidths=.005, annot=True, fmt='.2f')

        g.set_yticklabels(["phishing", "legitimate"], rotation=0)
        g.set_xticklabels(["phishing", "legitimate"], rotation=90)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()

        plt.savefig("{0}/confusion_matrix.png".format(save_to))
        print("{0}/confusion_matrix.png saved.".format(save_to))

        plt.close()

class Utils:

    def __init__(self):
        self.plotter = Plotter()

    def get_file_names(self, directory):
        file_names = sorted(["{}/{}".format(directory, line.strip()) for line in os.listdir("{}/".format(directory))])
        return file_names

    def save_results(self, params, test_acc, TEST_RESULTS, cm, report, dir_output):
        tm = str(datetime.datetime.now())[0:19].replace(" ", "_").replace(":", "-")
        tsm = tm.split("_")
        TEST_RESULTS['date'] = tsm[0]
        TEST_RESULTS['date_time'] = tsm[1]

        # TEST_RESULTS['epoch_history']['epoch_time'] = TEST_RESULTS['epoch_times']
        # TEST_RESULTS.pop('epoch_times')

        TEST_RESULTS['epoch'] = params['epoch']
        mlflow.log_param("epoch", params['epoch'])

        TEST_RESULTS['sequence_length'] = params['sequence_length']
        mlflow.log_param("sequence_length", params['sequence_length'])

        mlflow.log_param("epoch_train_duration", TEST_RESULTS['epoch_history']['train_duration'])
        mlflow.log_param("val_acc", TEST_RESULTS['epoch_history']['val_acc'])
        mlflow.log_param("train_duration", TEST_RESULTS['train_duration'])
        mlflow.log_param("train_loss", TEST_RESULTS['epoch_history']['loss'])
        mlflow.log_param("train_acc", TEST_RESULTS['epoch_history']['accuracy'])
        mlflow.log_param("test_acc", test_acc)

        mlflow.log_param("epoch_number", params['epoch'])
        mlflow.log_param("train_batch_size", params['batch_train'])
        mlflow.log_param("test_batch_size", params['batch_test'])
        mlflow.log_param("embed_dim", params['embedding_dimension'])

        open("{0}raw_test_results.json".format(dir_output), "w").write(json.dumps(TEST_RESULTS))
        mlflow.log_artifact("{0}raw_test_results.json".format(dir_output))

        open("{0}classification_report.txt".format(dir_output), "w").write(report)
        mlflow.log_artifact("{0}classification_report.txt".format(dir_output))

        self.plotter.plot_confusion_matrix(cm, save_to=dir_output)
        mlflow.log_artifact("{0}confusion_matrix.png".format(dir_output))

        # TEST_RESULTS['model_json'] = model_json
        mlflow.log_artifact("{0}ph_model".format(dir_output))

        self.plotter.plot_graphs(TEST_RESULTS['epoch_history']['accuracy'],
                                 TEST_RESULTS['epoch_history']['val_acc'],
                                 save_to=dir_output,
                                 name1="train_acc",
                                 name2="val_acc",
                                 figure_name="accuracy")
        mlflow.log_artifact("{0}accuracy.png".format(dir_output))

        """# saving embedding
        embeddings = model.layers[0].get_weights()[0]
        words_embeddings = {w: embeddings[idx].tolist() for w, idx in self.params['char_index'].items()}
        open("{0}char_embeddings.json".format(dir_output), "w").write(json.dumps(words_embeddings))"""