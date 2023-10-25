import os
import json
import time
import pprint
import argparse
import datetime
import numpy as np
import keras.callbacks as ckbs
from keras.utils import np_utils
from keras.models import Sequential
from keras.preprocessing import sequence
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
from keras import backend as K
from keras.layers import Dense, Dropout, LSTM, Bidirectional, Flatten, Conv1D, MaxPooling1D, Embedding, Input, GlobalMaxPooling1D, Convolution1D
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from dl_models import DlModels
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
# sunucuda calismak icin
plt.switch_backend('agg')

from keras.models import model_from_json


"""
  floyd run --gpu2 --data ebubekir/datasets/phishing_detection/1:dataset --message "cnn complex model bs: 7000" "python dl_floyd.py -arch cnn -ep 20 -bs 7000"
  floyd run --gpu2 --data ebubekir/datasets/phishing_detection/1:dataset --message "rnn complex model bs: 7000" "python dl_floyd.py -arch rnn -ep 20 -bs 7000"
  floyd run --gpu2 --data ebubekir/datasets/phishing_detection/1:dataset --message "brnn complex model bs: 7000" "python dl_floyd.py -arch brnn -ep 20 -bs 7000"
  floyd run --gpu2 --data ebubekir/datasets/phishing_detection/1:dataset --message "ann complex model bs: 7000" "python dl_floyd.py -arch ann -ep 20 -bs 7000"
  floyd run --gpu2 --data ebubekir/datasets/phishing_detection/1:dataset --message "att complex model bs: 7000" "python dl_floyd.py -arch att -ep 20 -bs 5000"
"""


pp = pprint.PrettyPrinter(indent=4)

TEST_RESULTS = {'data': {},
                "embedding": {},
                "hiperparameter": {},
                "test_result": {}}


class PhishingUrlDetection:

    def __init__(self):

        self.params = {'loss_function': 'binary_crossentropy',
                       'optimizer': 'adam',
                       'sequence_length': 200,
                       'batch_train': 5000,
                       'batch_test': 5000,
                       'categories': ['phishing', 'legitimate'],
                       'char_index': None,
                       'epoch': 30,
                       'embedding_dimension': 50,
                       'architecture': "cnn",
                       'result_dir': "/output/",
                       'dataset_dir': "../dataset/small_dataset/",
                       'char_embeddings': "../test_results/complex_cnn/cnn_complex_1/char_embeddings.json"}

        self.ml_plotter = Plotter()
        self.dl_models = DlModels(self.params['categories'], self.params['embedding_dimension'], self.params['sequence_length'])

    def set_params(self, args):
        self.params['test_case'] = args.test_case_name
        self.params['epoch'] = int(args.epoch)
        self.params['architecture'] = args.architecture
        self.params['batch_train'] = args.batch_size
        self.params['batch_test'] = args.batch_size

    def model_sum(self, x):
        try:
            TEST_RESULTS['hiperparameter']["model_summary"] += x
        except:
            TEST_RESULTS['hiperparameter']["model_summary"] = x

    def load_and_vectorize_data(self):
        print("load data")
        train = [line.strip() for line in open("{}/train.txt".format(self.params['dataset_dir']), "r").readlines()]
        test = [line.strip() for line in open("{}/test.txt".format(self.params['dataset_dir']), "r").readlines()]
        val = [line.strip() for line in open("{}/val.txt".format(self.params['dataset_dir']), "r").readlines()]

        TEST_RESULTS['data']['samples_train'] = len(train)
        TEST_RESULTS['data']['samples_test'] = len(test)
        TEST_RESULTS['data']['samples_val'] = len(val)
        TEST_RESULTS['data']['samples_overall'] = len(train) + len(test) + len(val)
        TEST_RESULTS['data']['name'] = self.params['dataset_dir']

        raw_x_train = [line.split("\t")[1] for line in train]
        raw_y_train = [line.split("\t")[0] for line in train]

        raw_x_val = [line.split("\t")[1] for line in val]
        raw_y_val = [line.split("\t")[0] for line in val]

        raw_x_test = [line.split("\t")[1] for line in test]
        raw_y_test = [line.split("\t")[0] for line in test]

        tokener = Tokenizer(lower=True, char_level=True, oov_token='-n-')
        tokener.fit_on_texts(raw_x_train + raw_x_val + raw_x_test)
        self.params['char_index'] = tokener.word_index
        print("train tokenizer")
        x_train = np.asanyarray(tokener.texts_to_sequences(raw_x_train))
        print("val tokenizer")
        x_val = np.asanyarray(tokener.texts_to_sequences(raw_x_val))
        print("test tokenizer")
        x_test = np.asanyarray(tokener.texts_to_sequences(raw_x_test))

        encoder = LabelEncoder()
        encoder.fit(self.params['categories'])

        y_train = np_utils.to_categorical(encoder.transform(raw_y_train), num_classes=len(self.params['categories']))
        y_val = np_utils.to_categorical(encoder.transform(raw_y_val), num_classes=len(self.params['categories']))
        y_test = np_utils.to_categorical(encoder.transform(raw_y_test), num_classes=len(self.params['categories']))

        return (x_train, y_train), (x_val, y_val), (x_test, y_test)

    def load_and_vectorize_data_for_ml(self):
        print("load data")
        train = [line.strip() for line in open("{}/train.txt".format(self.params['dataset_dir']), "r").readlines()]
        test = [line.strip() for line in open("{}/test.txt".format(self.params['dataset_dir']), "r").readlines()]
        val = [line.strip() for line in open("{}/val.txt".format(self.params['dataset_dir']), "r").readlines()]

        char_embeddings = json.loads(open(self.params['char_embeddings'], "r").read())

        TEST_RESULTS['data']['samples_train'] = len(train)
        TEST_RESULTS['data']['samples_test'] = len(test)
        TEST_RESULTS['data']['samples_val'] = len(val)
        TEST_RESULTS['data']['samples_overall'] = len(train) + len(test) + len(val)
        TEST_RESULTS['data']['name'] = self.params['dataset_dir']

        raw_x_train = [line.split("\t")[1] for line in train]
        raw_y_train = [line.split("\t")[0] for line in train]

        raw_x_val = [line.split("\t")[1] for line in val]
        raw_y_val = [line.split("\t")[0] for line in val]

        raw_x_test = [line.split("\t")[1] for line in test]
        raw_y_test = [line.split("\t")[0] for line in test]

        x_train = []
        x_val = []
        x_test = []

        print("file read is okey")

        for sample in raw_x_train:
            vector = []
            for s in sample:
                if len(vector) == 10000:
                    break
                else:
                    vector += char_embeddings[s]

            while len(vector) < 10000:
                vector.append(0.0)

            x_train.append(vector)

        print("train padding is okey")

        for sample in raw_x_test:
            vector = []
            for s in sample:
                if len(vector) == 10000:
                    break
                else:
                    vector += char_embeddings[s]

            while len(vector) < 10000:
                vector.append(0.0)

            x_test.append(vector)

        print("test padding is okey")

        for sample in raw_x_val:
            vector = []
            for s in sample:
                if len(vector) == 10000:
                    break
                else:
                    vector += char_embeddings[s]

            while len(vector) < 10000:
                vector.append(0.0)

            x_val.append(vector)

        print("val padding is okey")

        encoder = LabelEncoder()
        encoder.fit(self.params['categories'])

        x_train = np.asanyarray(x_train, dtype=np.float32)
        x_val = np.asanyarray(x_val, dtype=np.float32)
        x_test = np.asanyarray(x_test, dtype=np.float32)

        y_train = encoder.transform(raw_y_train)
        y_val = encoder.transform(raw_y_val)
        y_test = encoder.transform(raw_y_test)

        print("x_train shape: {} - y_train shape: {}\nx_val: {} - y_val: {}\nx_test:{} - y_test: {}".format(x_train.shape,
                                                                                                            len(y_train),
                                                                                                            x_val.shape,
                                                                                                            len(y_val),
                                                                                                            x_test.shape,
                                                                                                            len(y_test)))

        np.savez("x_train.npz", x_train)
        np.savez("x_test.npz", x_test)
        np.savez("x_val.npz", x_val)

        np.savez("y_train.npz", y_train)
        np.savez("y_test.npz", y_test)
        np.savez("y_val.npz", y_val)

        print("vektors saved.")

        return (x_train, y_train), (x_val, y_val), (x_test, y_test)

    def load_and_vectorize_data_for_ml2(self):
        print("load data")
        train = [line.strip() for line in open("{}/train.txt".format(self.params['dataset_dir']), "r").readlines()]
        test = [line.strip() for line in open("{}/test.txt".format(self.params['dataset_dir']), "r").readlines()]
        val = [line.strip() for line in open("{}/val.txt".format(self.params['dataset_dir']), "r").readlines()]

        char_embeddings = json.loads(open(self.params['char_embeddings'], "r").read())

        TEST_RESULTS['data']['samples_train'] = len(train)
        TEST_RESULTS['data']['samples_test'] = len(test)
        TEST_RESULTS['data']['samples_val'] = len(val)
        TEST_RESULTS['data']['samples_overall'] = len(train) + len(test) + len(val)
        TEST_RESULTS['data']['name'] = self.params['dataset_dir']

        raw_x_train = [line.split("\t")[1] for line in train]
        raw_y_train = [line.split("\t")[0] for line in train]

        raw_x_val = [line.split("\t")[1] for line in val]
        raw_y_val = [line.split("\t")[0] for line in val]

        raw_x_test = [line.split("\t")[1] for line in test]
        raw_y_test = [line.split("\t")[0] for line in test]

        x_train = []
        x_val = []
        x_test = []

        print("file read is okey")

        for sample in tqdm(raw_x_train):
            vector = []
            for s in sample:
                vector.append(char_embeddings[s])

            vv = np.asarray(vector)
            x_train.append(np.mean(vv, axis=0))

        print("train vec is okey")

        for sample in tqdm(raw_x_test):
            vector = []
            for s in sample:
                vector.append(char_embeddings[s])

            vv = np.asarray(vector)
            x_test.append(np.mean(vv, axis=0))

        print("test vec is okey")

        for sample in tqdm(raw_x_val):
            vector = []
            for s in sample:
                vector.append(char_embeddings[s])

            vv = np.asarray(vector)
            x_val.append(np.mean(vv, axis=0))

        print("val padding is okey")

        encoder = LabelEncoder()
        encoder.fit(self.params['categories'])

        x_train = np.asanyarray(x_train, dtype=np.float32)
        x_val = np.asanyarray(x_val, dtype=np.float32)
        x_test = np.asanyarray(x_test, dtype=np.float32)

        y_train = encoder.transform(raw_y_train)
        y_val = encoder.transform(raw_y_val)
        y_test = encoder.transform(raw_y_test)

        print("x_train shape: {} - y_train shape: {}\nx_val: {} - y_val: {}\nx_test:{} - y_test: {}".format(x_train.shape,
                                                                                                            len(y_train),
                                                                                                            x_val.shape,
                                                                                                            len(y_val),
                                                                                                            x_test.shape,
                                                                                                            len(y_test)))

        np.savez("x_train_mean.npz", x_train)
        np.savez("x_test_mean.npz", x_test)
        np.savez("x_val_mean.npz", x_val)

        np.savez("y_train_mean.npz", y_train)
        np.savez("y_test_mean.npz", y_test)
        np.savez("y_val_mean.npz", y_val)

        print("vektors saved.")

        return (x_train, y_train), (x_val, y_val), (x_test, y_test)

    def dl_algorithm(self, x_train, y_train, x_val, y_val, x_test, y_test):

        x_train = sequence.pad_sequences(x_train, maxlen=self.params['sequence_length'])
        x_test = sequence.pad_sequences(x_test, maxlen=self.params['sequence_length'])
        x_val = sequence.pad_sequences(x_val, maxlen=self.params['sequence_length'])

        print("train sequences: {}  |  test sequences: {} | val sequences: {}\n"
              "x_train shape: {}  |  x_test shape: {} | x_val shape: {}\n"
              "Building Model....".format(len(x_train), len(x_test), len(x_val), x_train.shape, x_test.shape, x_val.shape))

        # Build Deep Learning Architecture
        if self.params['architecture'] == "brnn":
            model = self.dl_models.brnn_complex(self.params['char_index'])
            TEST_RESULTS['hiperparameter']['architecture'] = "brnn"

        elif self.params['architecture'] == 'att':
            model = self.dl_models.att_complex(self.params['char_index'])
            TEST_RESULTS['hiperparameter']['architecture'] = "att"

        elif self.params['architecture'] == 'ann':
            model = self.dl_models.ann_complex(self.params['char_index'])
            TEST_RESULTS['hiperparameter']['architecture'] = "ann"

        elif self.params['architecture'] == 'rnn':
            model = self.dl_models.rnn_complex(self.params['char_index'])
            TEST_RESULTS['hiperparameter']['architecture'] = "rnn"

        elif self.params['architecture'] == 'cnn':
            model = self.dl_models.cnn_complex(self.params['char_index'])
            TEST_RESULTS['hiperparameter']['architecture'] = "cnn"

        elif self.params['architecture'] == 'cnn2':
            model = self.dl_models.cnn_complex2(self.params['char_index'])
            TEST_RESULTS['hiperparameter']['architecture'] = "cnn"

        elif self.params['architecture'] == 'cnn3':
            model = self.dl_models.cnn_complex3(self.params['char_index'])
            TEST_RESULTS['hiperparameter']['architecture'] = "cnn"


        # Build Deep Learning Architecture

        model.compile(loss=self.params['loss_function'], optimizer=self.params['optimizer'], metrics=['accuracy'])

        model.summary()
        model.summary(print_fn=lambda x: self.model_sum(x + '\n'))

        hist = model.fit(x_train, y_train,
                         batch_size=self.params['batch_train'],
                         epochs=self.params['epoch'],
                         shuffle=True,
                         validation_data=(x_val, y_val),
                         callbacks=[CustomCallBack()])

        t = time.time()
        score, acc = model.evaluate(x_test, y_test, batch_size=self.params['batch_test'])

        TEST_RESULTS['test_result']['test_time'] = time.time() - t

        y_test = list(np.argmax(np.asanyarray(np.squeeze(y_test), dtype=int).tolist(), axis=1))
        y_pred = model.predict_classes(x_test, batch_size=self.params['batch_test'], verbose=1).tolist()
        report = classification_report(y_test, y_pred, target_names=self.params['categories'])
        print(report)
        TEST_RESULTS['test_result']['report'] = report
        TEST_RESULTS['epoch_history'] = hist.history
        TEST_RESULTS['test_result']['test_acc'] = acc
        TEST_RESULTS['test_result']['test_loss'] = score

        test_confusion_matrix = confusion_matrix(y_test, y_pred)
        #TEST_RESULTS['test_result']['y_test'] = y_test
        #TEST_RESULTS['test_result']['y_prediction'] = y_predicted
        TEST_RESULTS['test_result']['test_confusion_matrix'] = test_confusion_matrix.tolist()

        print('Test loss: {0}  |  test accuracy: {1}'.format(score, acc))
        self.save_results(model)

    def traditional_ml(self, arch):

        x_train = np.load("x_train_mean.npz")['arr_0']
        y_train = np.load("y_train_mean.npz")['arr_0']
        start_time = time.time()

        if arch == "NB":
            print("traditional ML - NB is running")
            gnb = GaussianNB()
            model = gnb.fit(x_train, y_train)
        elif arch == "RF":
            print("traditional ML - RF is running")
            clf = RandomForestClassifier(n_estimators=10, random_state=0, verbose=1, n_jobs=64)
            model = clf.fit(x_train, y_train)
        elif arch == "SVM":
            print("traditional ML - SVM is running")
            clf = svm.SVC(gamma='scale', verbose=True)
            model = clf.fit(x_train, y_train)

        elif arch == "KNN":
            print("traditional ML - KNN is running")
            clf = KNeighborsClassifier(n_neighbors=3, metric='minkowski', p=2, n_jobs=64)
            model = clf.fit(x_train, y_train)

        elif arch == "LR":
            print("traditional ML - LR is running")
            clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial', verbose=1, n_jobs=64)
            model = clf.fit(x_train, y_train)

        TEST_RESULTS['train_duration'] = time.time() - start_time

        print("x_train shape: {} - y_train shape: {}".format(x_train.shape, len(y_train)))

        x_train = None

        x_val = np.load("x_val_mean.npz")['arr_0']
        y_val = np.load("y_val_mean.npz")['arr_0']

        x_test = np.load("x_test_mean.npz")['arr_0']
        y_test = np.load("y_test_mean.npz")['arr_0']

        st = time.time()
        model_prediction_val = model.predict(x_val)
        TEST_RESULTS['val_duration'] = time.time() - st

        st = time.time()
        model_prediction_test = model.predict(x_test)
        TEST_RESULTS['test_duration'] = time.time() - st

        # model_probability = model.predict_proba(x_val)
        acc_val = accuracy_score(y_val, model_prediction_val)
        acc_test = accuracy_score(y_test, model_prediction_test)

        report_val = classification_report(y_val, model_prediction_val, target_names=self.params['categories'])
        report_test = classification_report(y_test, model_prediction_test, target_names=self.params['categories'])
        val_confusion_matrix = confusion_matrix(y_val, model_prediction_val)
        test_confusion_matrix = confusion_matrix(y_test, model_prediction_test)

        TEST_RESULTS['val_confusion_matrix'] = val_confusion_matrix.tolist()
        TEST_RESULTS['test_confusion_matrix'] = test_confusion_matrix.tolist()

        TEST_RESULTS['test_acc'] = acc_test
        TEST_RESULTS['test_val'] = acc_val

        TEST_RESULTS['test_report'] = report_test
        TEST_RESULTS['val_report'] = report_val

        print(report_val)
        print(val_confusion_matrix)

        open("../result/{}_raw_test_results.json".format(arch), "w").write(json.dumps(TEST_RESULTS))
        open("../result/{}_classification_report_test.txt".format(arch), "w").write(str(TEST_RESULTS['test_report']))
        open("../result/{}_classification_report_val.txt".format(arch), "w").write(str(TEST_RESULTS['val_report']))
        self.ml_plotter.plot_confusion_matrix_2(test_confusion_matrix, self.params['categories'], save_to="../result/test_{}_norm_cm.txt".format(arch))
        self.ml_plotter.plot_confusion_matrix_2(val_confusion_matrix, self.params['categories'], save_to="../result/val_{}_norm_cm.txt".format(arch), normalized=True)

    def save_results(self, model):
        tm = str(datetime.datetime.now())[0:19].replace(" ", "_").replace(":", "-")
        tsm = tm.split("_")
        TEST_RESULTS['date'] = tsm[0]
        TEST_RESULTS['date_time'] = tsm[1]
        TEST_RESULTS['test_case'] = self.params['test_case']

        TEST_RESULTS['embedding']['vocabulary_size'] = len(self.params['char_index'])
        TEST_RESULTS["embedding"]['embedding_dimension'] = self.params['embedding_dimension']

        TEST_RESULTS['epoch_history']['epoch_time'] = TEST_RESULTS['epoch_times']
        TEST_RESULTS.pop('epoch_times')

        TEST_RESULTS['hiperparameter']['epoch'] = self.params['epoch']
        TEST_RESULTS['hiperparameter']['train_batch_size'] = self.params['batch_train']
        TEST_RESULTS['hiperparameter']['test_batch_size'] = self.params['batch_test']
        TEST_RESULTS['hiperparameter']['sequence_length'] = self.params['sequence_length']

        TEST_RESULTS['params'] = self.params

        model_json = model.to_json()
        model.save("{}model_all.h5".format(self.params['result_dir']))
        open("{0}model.json".format(self.params['result_dir']), "w").write(json.dumps(model_json))
        model.save_weights("{0}weights.h5".format(self.params['result_dir']))

        open("{0}raw_test_results.json".format(self.params['result_dir']), "w").write(json.dumps(TEST_RESULTS))
        open("{0}model_summary.txt".format(self.params['result_dir']), "w").write(TEST_RESULTS['hiperparameter']["model_summary"])
        open("{0}classification_report.txt".format(self.params['result_dir']), "w").write(TEST_RESULTS['test_result']['report'])

        self.ml_plotter.plot_graphs(TEST_RESULTS['epoch_history']['acc'], TEST_RESULTS['epoch_history']['val_acc'], save_to=self.params['result_dir'], name="accuracy")
        self.ml_plotter.plot_graphs(TEST_RESULTS['epoch_history']['loss'], TEST_RESULTS['epoch_history']['val_loss'], save_to=self.params['result_dir'], name="loss")
        self.ml_plotter.plot_confusion_matrix(TEST_RESULTS['test_result']['test_confusion_matrix'], self.params['categories'], save_to=self.params['result_dir'])
        self.ml_plotter.plot_confusion_matrix(TEST_RESULTS['test_result']['test_confusion_matrix'], self.params['categories'], save_to=self.params['result_dir'], normalized=True)

        # saving embedding
        embeddings = model.layers[0].get_weights()[0]
        words_embeddings = {w: embeddings[idx].tolist() for w, idx in self.params['char_index'].items()}
        open("{0}char_embeddings.json".format(self.params['result_dir']), "w").write(json.dumps(words_embeddings))


class CustomCallBack(ckbs.Callback):

    def __init__(self):
        ckbs.Callback.__init__(self)
        TEST_RESULTS['epoch_times'] = []

    def on_epoch_begin(self, batch, logs=None):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        TEST_RESULTS['epoch_times'].append(time.time() - self.epoch_time_start)


class Plotter:

    def plot_graphs(self, train, val, save_to=None, name="accuracy"):

        if name == "accuracy":
            val, = plt.plot(val, label="val_acc")
            train, = plt.plot(train, label="train_acc")
        else:
            val, = plt.plot(val, label="val_loss")
            train, = plt.plot(train, label="train_loss")

        plt.ylabel(name)
        plt.xlabel("epoch")

        plt.legend(handles=[val, train], loc=2)

        if save_to:
            plt.savefig("{0}/{1}.png".format(save_to, name))

        plt.close()

    def plot_confusion_matrix(self, confusion_matrix, categories, save_to=None, normalized=False):

        sns.set()
        plt.rcParams.update({'font.size': 14})
        plt.figure(figsize=(14.0, 7.0))

        if normalized:
            row_sums = np.asanyarray(confusion_matrix).sum(axis=1)
            matrix = confusion_matrix / row_sums[:, np.newaxis]
            matrix = [line.tolist() for line in matrix]
            g = sns.heatmap(matrix, annot=True, fmt='f', xticklabels=True, yticklabels=True)

        else:
            matrix = confusion_matrix
            g = sns.heatmap(matrix, annot=True, fmt='d', xticklabels=True, yticklabels=True)

        g.set_yticklabels(categories, rotation=0)
        g.set_xticklabels(categories, rotation=90)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()

        if save_to:
            if normalized:
                plt.savefig("{0}/{1}.png".format(save_to, "normalized_confusion_matrix"))
            else:
                plt.savefig("{0}/{1}.png".format(save_to, "confusion_matrix"))

    def plot_confusion_matrix_2(self, confusion_matrix, categories, save_to=None, normalized=False):

        sns.set()
        plt.rcParams.update({'font.size': 14})
        plt.figure(figsize=(14.0, 7.0))

        if normalized:
            row_sums = np.asanyarray(confusion_matrix).sum(axis=1)
            matrix = confusion_matrix / row_sums[:, np.newaxis]
            matrix = [line.tolist() for line in matrix]
            g = sns.heatmap(matrix, annot=True, fmt='f', xticklabels=True, yticklabels=True)

        else:
            matrix = confusion_matrix
            g = sns.heatmap(matrix, annot=True, fmt='d', xticklabels=True, yticklabels=True)

        g.set_yticklabels(categories, rotation=0)
        g.set_xticklabels(categories, rotation=90)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()

        if save_to:
            if normalized:
                plt.savefig("{0}.png".format(save_to))
            else:
                plt.savefig("{0}.png".format(save_to))


def argument_parsing():
    parser = argparse.ArgumentParser()
    parser.add_argument("-ep", "--epoch", default=10, help='The number of epoch')
    parser.add_argument("-nm", "--test_case_name", help='Test Case Name')
    parser.add_argument("-arch", "--architecture", default="cnn", help='Architecture to be tested')
    parser.add_argument("-bs", "--batch_size", default=1000, help='batch size', type=int)

    args = parser.parse_args()

    return args


def main():

    args = argument_parsing()
    vc = PhishingUrlDetection()
    vc.set_params(args)

    # (x_train, y_train), (x_val, y_val), (x_test, y_test) = vc.load_and_vectorize_data()
    # vc.dl_algorithm(x_train, y_train, x_val, y_val, x_test, y_test)

    #(x_train, y_train), (x_val, y_val), (x_test, y_test) = vc.load_and_vectorize_data_for_ml2()
    vc.traditional_ml(args.architecture)


if __name__ == '__main__':
    main()
