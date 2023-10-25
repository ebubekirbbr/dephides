import os
import json
import subprocess
import sys
import time
import pprint
import argparse
import datetime
import numpy as np
import seaborn as sns
from sys import stdout
from tqdm import tqdm
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras.preprocessing import sequence
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
import mlflow
from dl_models import DlModels
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow import keras
from utils import Utils, Plotter, PhishingDataGen
import tensorflow as tf
from keras import backend as K

# sunucuda calismak icin
plt.switch_backend('agg')

pp = pprint.PrettyPrinter(indent=4)

class PhishingUrlDetection:

    def __init__(self, epoch):

        self.params = {'optimizer': 'adam',
                       'sequence_length': 512,
                       'batch_train': 5000,
                       'batch_test': 5000,
                       'categories': ['phishing', 'legitimate'],
                       'char_index': None,
                       'epoch': epoch,
                       'embedding_dimension': 100,
                       'result_dir': "../result/",
                       'dataset_dir': "../dataset/big_dataset",
                       "train_dir": "../dataset/big_dataset/train",
                       "test_dir": "../dataset/big_dataset/test",
                       "val_dir": "../dataset/big_dataset/val",
                       }

        accepted_chars = " !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[]\\^_`abcdefghijklmnopqrstuvwxyz{|}~"

        self.tokener = Tokenizer(lower=True, char_level=True, oov_token='-n-')
        self.tokener.word_index = json.loads(open("../dataset/char_index").read())
        self.utils = Utils()

        if not os.path.exists(self.params['result_dir']):
            os.mkdir(self.params['result_dir'])
            print("Directory ", self.params['result_dir'], " Created ")
        else:
            print("Directory ", self.params['result_dir'], " already exists")

        self.ml_plotter = Plotter()
        self.dl_models = DlModels(self.params['categories'],
                                  self.params['embedding_dimension'],
                                  self.params['sequence_length'])

    def dl_val(self, model, test_gen):

        accs = []
        y_true_all = []
        y_pred_all = []
        t = time.time()
        for j, data in tqdm(enumerate(test_gen), desc="batch"):
            x = data[0]  # vec
            y = data[1]  # label

            # y_pred = model.predict_on_batch(x)
            y_pred = model.predict(x)
            y_pred = np.where(np.asanyarray(y_pred) > 0.5, 1, 0)

            y_true_all += y.tolist()
            y_pred_all += y_pred.tolist()

        y_pred_all = np.asanyarray(y_pred_all)
        y_true_all = np.asanyarray(y_true_all)

        custom_acc = accuracy_score(y_true_all, y_pred_all)
        conf_matrix = confusion_matrix(y_true_all, y_pred_all)
        report = classification_report(y_true_all, y_pred_all)

        return round(custom_acc, 4), report, conf_matrix, time.time() - t

    def dl_algorithm(self):

        TEST_RESULTS = {'data': {"train_gen": 0, "test_gen": 0},
                        "custom_acc": None,
                        "train_duration": None,
                        "epoch_history": {"train_duration": [],
                                          "val_duration": [],
                                          "accuracy": [],
                                          "loss": [],
                                          'val_acc': []}}

        file_names_train = self.utils.get_file_names(self.params["train_dir"])
        file_names_test = self.utils.get_file_names(self.params["test_dir"])
        file_names_val = self.utils.get_file_names(self.params["val_dir"])

        train_gen = PhishingDataGen(file_names_train, 1)
        test_gen = PhishingDataGen(file_names_test, 1)
        val_gen = PhishingDataGen(file_names_val, 1)

        print("train_gen: {} - test_gen: {} - val_gen: {}".format(len(train_gen), len(test_gen), len(val_gen)))

        model = self.dl_models.cnn_complex3(self.tokener.word_index)
        mlflow.log_param('architecture', "cnn3")
        # Build Deep Learning Architecture

        model.compile(loss="binary_crossentropy", optimizer=self.params['optimizer'], metrics=['accuracy'])

        model.summary()

        loss = None
        acc = None
        val_custom_acc = None
        val_report = None
        val_confusion_matrix_result = None
        val_conf_matrix = None

        start = time.time()

        for i in range(self.params["epoch"]):
            t = time.time()

            for j, data in enumerate(train_gen):
                x = data[0] #vec
                y = data[1] #label

                res = model.train_on_batch(x=x, y=y, reset_metrics=False)

                loss = res[0]
                acc = res[1]
                print("epoch:{}/{} | batch: {}/{} | duration: {} sec | loss:{} | acc:{}".format(i, self.params["epoch"], j, len(train_gen), int(time.time() - t), round(loss, 4), round(acc, 4)), end="\r")

            TEST_RESULTS["epoch_history"]["loss"].append(loss)
            TEST_RESULTS["epoch_history"]["accuracy"].append(acc)
            TEST_RESULTS["epoch_history"]["train_duration"].append(time.time() - t)
            print("\n")
            val_custom_acc, val_report, val_conf_matrix, val_duration = self.dl_val(model, val_gen)
            TEST_RESULTS["epoch_history"]["val_duration"].append(val_duration)
            TEST_RESULTS["epoch_history"]["val_acc"].append(val_custom_acc)
            print("epoch:{}/{} | len_train_gen: {} | duration: {} sec | loss:{} | acc:{} - val_acc: {}".format(i, self.params["epoch"], len(train_gen), int(time.time() - t), round(loss, 4), round(acc, 4), val_custom_acc))

        TEST_RESULTS["train_duration"] = time.time() - start
        test_custom_acc, test_report, test_conf_matrix, test_duration = self.dl_val(model, test_gen)

        #tf.saved_model.save(model, "{}ph_model".format(self.params["result_dir"]))
        subprocess.call("rm -fr ../result/ph_model", shell=True)
        tf.keras.models.save_model(model, "{}ph_model".format(self.params["result_dir"]))

        """keras.models.save_model(
            model,
            "{}ph_model".format(self.params["result_dir"]),
            overwrite=True,
            include_optimizer=True,
            save_format=None,
            signatures=None,
            options=None
        )"""

        self.utils.save_results(self.params, test_custom_acc, TEST_RESULTS, test_conf_matrix, test_report, self.params["result_dir"])

    def test_model(self):

        #model = tf.saved_model.load(export_dir="{}/saved_model".format(self.params["result_dir"]), tags="serve")
        model = keras.models.load_model("{}ph_model".format(self.params["result_dir"]))
        print("model read")
        urls = ["gobyexample.com/login/ender/phishing_url_detection/merge_requests/new?merge_request",
                "gobyexample.com/login",
                "toksir.com/home/rs/ender/phishing_url_detection/src/mlruns/1/35354c940c3b48d08237381c4b4d5f92/artifacts/ph_model",
                "session-4154179783.nationalcity.com.userpro.tw/corporate/onlineservices/TreasuryMgmt/".lower(),
                "gitlab.roksit.com/ender/phishing_url_detection/merge_requests/new?merge_request%5Bsource_branch%5D=dev"] * 20000

        print("data loading")
        #data = self.rks.get_all_data_paralel({"query": {"match_all": {}}, "size": 10000}, "phishing_urls_v02", es, part=10)
        print("data loaded")
        #urls = [sample["_source"]["url"] for sample in data]
        vec = self.tokener.texts_to_sequences(urls)
        vec = sequence.pad_sequences(vec, maxlen=512)
        print(vec.shape)
        t = time.time()

        res = model.predict(vec)
        for i, line in enumerate(res):
            #print("{} - {}".format(round(float(line[0]), 4), urls[i]))
            pass

        print("time: {} sn".format(time.time()- t))


def argument_parsing():
    parser = argparse.ArgumentParser()
    parser.add_argument("-ep", "--epoch", default=10, help='The number of epoch', type=int)
    parser.add_argument("-arch", "--architecture", default="cnn", help='Architecture to be tested')
    parser.add_argument("-t", "--test", action='store_true', help='test model')
    parser.add_argument("-bs", "--batch_size", default=2000, help='batch size')

    args = parser.parse_args()

    return args


def main():

    args = argument_parsing()
    vc = PhishingUrlDetection(args.epoch)

    if args.test:
        vc.test_model()
    else:
        mlflow.start_run(experiment_id=1)
        vc.dl_algorithm()
        mlflow.end_run()


if __name__ == '__main__':
    main()
