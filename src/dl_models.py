from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Bidirectional, Flatten, Conv1D, MaxPooling1D, Embedding, Input, GlobalMaxPooling1D, Convolution1D
#from keras_self_attention import SeqSelfAttention


class DlModels:

    def __init__(self, categories, embed_dim, sequence_length):

        self.categories = categories
        self.embed_dim = embed_dim
        self.sequence_length = sequence_length

    def rnn_base(self, char_index):

        model = Sequential()
        voc_size = len(char_index.keys())

        model.add(Embedding(voc_size + 1, self.embed_dim))
        model.add(LSTM(128))
        model.add(Dense(len(self.categories), activation='sigmoid'))

        return model

    def brnn_base(self, char_index):

        model = Sequential()
        voc_size = len(char_index.keys())

        model.add(Embedding(voc_size + 1, self.embed_dim))
        model.add(Bidirectional(LSTM(128)))
        model.add(Dense(len(self.categories), activation='sigmoid'))

        return model

    def cnn_base(self, char_index):

        model = Sequential()
        voc_size = len(char_index.keys())
        model.add(
            Embedding(voc_size + 1, self.embed_dim, input_length=self.sequence_length))
        model.add(Convolution1D(128, 3, activation='tanh'))
        model.add(Flatten())

        model.add(Dense(len(self.categories), activation='sigmoid'))

        return model

    def ann_base(self, char_index):

        model = Sequential()
        voc_size = len(char_index.keys())

        model.add(
            Embedding(voc_size + 1, self.embed_dim, input_length=self.sequence_length))
        model.add(Dense(128, activation='tanh'))
        model.add(Flatten())

        model.add(Dense(len(self.categories), activation='sigmoid'))

        return model

    def att_base(self, char_index):

        model = Sequential()
        voc_size = len(char_index.keys())

        model.add(Embedding(voc_size + 1, self.embed_dim, input_length=self.sequence_length))
        # model.add(Bidirectional(LSTM(units=128, return_sequences=True)))
        model.add(SeqSelfAttention(attention_activation='sigmoid'))
        model.add(Flatten())

        model.add(Dense(len(self.categories), activation='sigmoid'))

        return model

    def rnn_complex(self, char_index):

        model = Sequential()
        voc_size = len(char_index.keys())
        print("voc_size: {}".format(voc_size))

        model.add(Embedding(voc_size + 1, self.embed_dim))
        model.add(LSTM(128, return_sequences=True))

        model.add(LSTM(128, return_sequences=True))

        model.add(LSTM(128, return_sequences=True))

        model.add(LSTM(128, return_sequences=True))

        model.add(LSTM(128, return_sequences=True))

        model.add(LSTM(128, return_sequences=True))

        model.add(LSTM(128))

        model.add(Dense(len(self.categories), activation='sigmoid'))

        return model

    def brnn_complex(self, char_index):

        model = Sequential()
        voc_size = len(char_index.keys())
        print("voc_size: {}".format(voc_size))
        model.add(Embedding(voc_size + 1, self.embed_dim))
        model.add(Bidirectional(LSTM(64, return_sequences=True)))

        model.add(Bidirectional(LSTM(64, return_sequences=True)))

        model.add(Bidirectional(LSTM(64, return_sequences=True)))

        model.add(Bidirectional(LSTM(64, return_sequences=True)))

        model.add(Bidirectional(LSTM(64, return_sequences=True)))

        model.add(Bidirectional(LSTM(64, return_sequences=True)))

        model.add(Bidirectional(LSTM(128)))

        model.add(Dense(len(self.categories), activation='sigmoid'))

        return model

    def ann_complex(self, char_index):

        model = Sequential()
        voc_size = len(char_index.keys())
        print("voc_size: {}".format(voc_size))

        model.add(Embedding(voc_size + 1, self.embed_dim, input_length=self.sequence_length))

        model.add(Dense(128, activation='tanh'))

        model.add(Dense(128, activation='tanh'))

        model.add(Dense(128, activation='tanh'))

        model.add(Dense(128, activation='tanh'))

        model.add(Dense(128, activation='tanh'))

        model.add(Dense(128, activation='tanh'))

        model.add(Dense(128, activation='tanh'))
        model.add(Flatten())

        model.add(Dense(len(self.categories), activation='sigmoid'))

        return model

    def att_complex(self, char_index):

        model = Sequential()
        voc_size = len(char_index.keys())

        model.add(Embedding(voc_size + 1, self.embed_dim, input_length=self.sequence_length))

        model.add(LSTM(units=128, return_sequences=True))

        model.add(Bidirectional(LSTM(units=128, return_sequences=True)))

        model.add(SeqSelfAttention(attention_activation='sigmoid'))

        model.add(Bidirectional(LSTM(units=128, return_sequences=True)))

        model.add(LSTM(units=128, return_sequences=True))

        model.add(SeqSelfAttention(attention_activation='sigmoid'))

        model.add(LSTM(units=128, return_sequences=True))

        model.add(Flatten())

        model.add(Dense(len(self.categories), activation='sigmoid'))

        return model

    def cnn_complex(self, char_index):

        model = Sequential()
        voc_size = len(char_index.keys())
        print("voc_size: {}".format(voc_size))
        model.add(Embedding(voc_size + 1, self.embed_dim, input_length=self.sequence_length))

        model.add(Convolution1D(128, 3, activation='tanh'))
        model.add(MaxPooling1D(3))
        model.add(Dropout(0.2))

        model.add(Convolution1D(128, 7, activation='tanh', padding='same'))
        model.add(Dropout(0.2))

        model.add(Convolution1D(128, 5, activation='tanh', padding='same'))
        model.add(Dropout(0.2))

        model.add(Convolution1D(128, 3, activation='tanh', padding='same'))
        model.add(MaxPooling1D(3))
        model.add(Dropout(0.2))

        model.add(Convolution1D(128, 5, activation='tanh', padding='same'))
        model.add(Dropout(0.2))

        model.add(Convolution1D(128, 3, activation='tanh', padding='same'))
        model.add(MaxPooling1D(3))
        model.add(Dropout(0.2))

        model.add(Convolution1D(128, 3, activation='tanh', padding='same'))
        model.add(MaxPooling1D(3))
        model.add(Dropout(0.2))

        model.add(Flatten())

        model.add(Dense(len(self.categories), activation='sigmoid'))

        return model

    def cnn_complex2(self, char_index):
        model = Sequential()
        voc_size = len(char_index.keys())
        print("voc_size: {}".format(voc_size))
        model.add(Embedding(voc_size + 1, self.embed_dim, input_length=self.sequence_length))

        model.add(Convolution1D(128, 3, activation='tanh'))
        model.add(MaxPooling1D(3))
        model.add(Dropout(0.2))

        model.add(Convolution1D(256, 7, activation='tanh', padding='same'))
        model.add(Dropout(0.2))

        model.add(Convolution1D(96, 5, activation='tanh', padding='same'))
        model.add(Dropout(0.2))

        model.add(Convolution1D(128, 3, activation='tanh', padding='same'))
        model.add(Dropout(0.2))

        model.add(Convolution1D(196, 5, activation='tanh', padding='same'))
        model.add(Dropout(0.2))

        model.add(Convolution1D(128, 3, activation='tanh', padding='same'))
        model.add(Dropout(0.2))

        model.add(Convolution1D(96, 5, activation='tanh', padding='same'))
        model.add(MaxPooling1D(3))
        model.add(Dropout(0.2))

        model.add(Convolution1D(128, 3, activation='tanh', padding='same'))
        model.add(Dropout(0.2))

        model.add(Convolution1D(196, 5, activation='tanh', padding='same'))
        model.add(Dropout(0.2))

        model.add(Convolution1D(128, 7, activation='tanh', padding='same'))
        model.add(Dropout(0.2))

        model.add(Convolution1D(96, 3, activation='tanh', padding='same'))
        model.add(MaxPooling1D(3))
        model.add(Dropout(0.2))

        model.add(Flatten())

        model.add(Dense(len(self.categories), activation='sigmoid'))

        return model

    def cnn_complex3(self, char_index):
        model = Sequential()
        voc_size = len(char_index.keys())
        print("voc_size: {}".format(voc_size))
        model.add(Embedding(voc_size + 1, self.embed_dim, input_length=self.sequence_length, name="inputLayer"))

        model.add(Convolution1D(128, 3, activation='tanh'))
        model.add(MaxPooling1D(3))

        model.add(Convolution1D(256, 7, activation='tanh', padding='same'))
        model.add(Convolution1D(96, 5, activation='tanh', padding='same'))
        model.add(Convolution1D(128, 3, activation='tanh', padding='same'))
        model.add(MaxPooling1D(3))
        model.add(Convolution1D(196, 5, activation='tanh', padding='same'))
        model.add(Convolution1D(128, 3, activation='tanh', padding='same'))
        model.add(Convolution1D(96, 5, activation='tanh', padding='same'))
        model.add(Convolution1D(128, 3, activation='tanh', padding='same'))
        model.add(Convolution1D(196, 5, activation='tanh', padding='same'))
        model.add(Convolution1D(128, 7, activation='tanh', padding='same'))
        model.add(Convolution1D(96, 3, activation='tanh', padding='same'))
        model.add(MaxPooling1D(3))

        model.add(Convolution1D(196, 5, activation='tanh', padding='same'))
        model.add(Convolution1D(128, 7, activation='tanh', padding='same'))
        model.add(MaxPooling1D(3))

        model.add(Convolution1D(196, 5, activation='tanh', padding='same'))
        model.add(Convolution1D(128, 7, activation='tanh', padding='same'))
        model.add(Convolution1D(96, 3, activation='tanh', padding='same'))

        model.add(Flatten())

        model.add(Dense(1, activation='sigmoid', name="inferenceLayer"))

        return model

    def custom_model(self, char_index):

        model = None

        return model
