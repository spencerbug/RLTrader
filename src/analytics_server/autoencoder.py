# autoencoder.py
#  defines a autoencoder transformer model
# which perofrms autoregressive forecasting on time series data
# architecture: time series data -> [encoder] -> embedding -> [decoder] -> time series data (+1 forecast step)
# the forecast and the embedding are passed to the trading agents
# procedure: instantiate TimeSeriesAutoencoder, load weights, pretrain on data (optional), then run train_and_predict in generator pipeline

import tensorflow as tf
from keras.layers import Input, Dense, Dropout, LayerNormalization
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from collections import deque
import numpy as np
import random
from threading import Thread, Lock
from queue import Queue
import time


"""
Time Series Autoencoder
@param input_seq_len: length of input sequence
@param output_seq_len: length of the output sequence ()
"""
class TimeSeriesAutoencoder:
    def __init__(self, input_seq_len, output_seq_len, d_model, n_heads, n_layers, ff_dim, dropout_rate, patience):
        self.input_seq_len = input_seq_len
        self.output_seq_len = output_seq_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate
        self.patience = patience

        self.encoder_inputs = Input(shape=(input_seq_len, 1))
        self.decoder_inputs = Input(shape=(output_seq_len - 1, 1))

        self.encoder = self.create_encoder()
        self.decoder = self.create_decoder()

        self.autoencoder = self.create_autoencoder()
    
    def create_encoder_layer(self, layer_num):
        seq_len, d_model, n_heads, ff_dim, dropout_rate = self.input_seq_len, self.d_model, self.n_heads, self.ff_dim, self.dropout_rate

        # Multi-head self attention layer
        multi_head_att = tf.keras.layers.MultiHeadAttention(
            key_dim=d_model,
            num_heads=n_heads,
            dropout=dropout_rate,
            name=f"EncoderLayer_{layer_num}_MultiHeadSelfAttention"
        )
        att_ouputs = multi_head_att(self.encoder_inputs, self.encoder_inputs)
        att_outputs = LayerNormalization()(att_outputs)
        att_outputs = Dropout(dropout_rate)(att_outputs)
        att_outputs = tf.keras.layers.Add()([self.encoder_inputs, att_outputs])

        # Feed forward layer
        feed_forward = tf.keras.Sequential([
            Dense(ff_dim, activation="relu"),
            Dense(d_model)
        ])
        ff_outputs = feed_forward(att_outputs)
        ff_outputs = LayerNormalization()(ff_outputs)
        ff_outputs = Dropout(dropout_rate)(ff_outputs)
        ff_outputs = tf.keras.layers.Add()([att_outputs, ff_outputs])

        return Model(self.encoder_inputs, ff_outputs, name=f"Encoder_Layer_{layer_num}")
    
    def create_decoder_layer(self, layer_num):
        seq_len, d_model, n_heads, ff_dim, dropout_rate = self.output_seq_len - 1, self.d_model, self.n_heads, self.ff_dim, self.dropout_rate

        # Masked multi-head self-attention layer
        masked_multi_head_att = tf.keras.layers.MultiHeadAttention(
            key_dim=d_model // n_heads,
            num_heads=n_heads,
            dropout=dropout_rate,
            name=f"DecoderLayer_{layer_num}_MultiHeadSelfAttention"
        )
        att_outputs = masked_multi_head_att(self.decoder_inputs, self.decoder_inputs)
        att_outputs = LayerNormalization()(att_outputs)
        att_outputs = Dropout(dropout_rate)(att_outputs)
        att_outputs = tf.keras.layers.Add()([self.decoder_inputs, att_outputs])

        # Multi-head attention layer with encoder outputs as inputs
        multi_head_att = tf.keras.layers.MultiHeadAttention(
            key_dim=d_model // n_heads,
            num_heads=n_heads,
            dropout=dropout_rate,
            name=f"DecoderLayer_{layer_num}_MultiHeadAttention"
        )

        encoder_outputs = Input(shape=(self.input_seq_len, self.d_model))
        att_outputs2 = multi_head_att([att_outputs, encoder_outputs], [encoder_outputs, encoder_outputs])
        att_outputs2 = LayerNormalization()(att_outputs2)
        att_outputs2 = Dropout(dropout_rate)(att_outputs2)
        att_outputs2 = tf.keras.layers.Add()([att_outputs, att_outputs2])

        # Feed forward layer
        feed_forward = tf.keras.Sequential([
            Dense(ff_dim, activation="relu"),
            Dense(d_model)
        ])
        ff_outputs = feed_forward(att_outputs2)
        ff_outputs = LayerNormalization()(ff_outputs)
        ff_outputs = Dropout(dropout_rate)(ff_outputs)
        ff_outputs = tf.keras.layers.Add()([att_outputs2, ff_outputs])
        
        return Model([self.decoder_inputs, encoder_outputs], ff_outputs, name=f"Decoder_Layer_{layer_num}")

    def create_encoder(self):
        encoder_inputs = self.encoder_inputs

        # encoder layers
        for i in range(self.n_layers):
            encoder_outputs = self.create_encoder_layer(i)(encoder_inputs)
            encoder_inputs = encoder_outputs
        
        return Model(self.encoder_inputs, encoder_outputs, name="Encoder")
    
    def create_decoder(self):
        decoder_inputs = self.decoder_inputs
        encoder_outputs = Input(shape=(self.input_seq_len, self.d_model))

        # decoder layers
        for i in range(self.n_layers):
            decoder_outputs = self.create_decoder_layer(i)([decoder_inputs, encoder_outputs])
            decoder_inputs = decoder_outputs
        
        # Final sense layer to output predictions
        decoder_outputs = Dense(1)(decoder_outputs)
        return Model([self.decoder_inputs, encoder_outputs], decoder_outputs, name="Decoder")
    
    def create_autoencoder(self):
        # Encoder
        encoder_inputs = Input(shape=(self.input_seq_len, self.d_model), name="Encoder_Input")
        encoder_outputs = self.create_encoder()(encoder_inputs)

        # Decoder
        decoder_inputs = Input(shape=(self.output_seq_len - 1, self.d_model), name="Decoder_Input")
        decoder_outputs = self.create_decoder()([decoder_inputs, encoder_outputs])

        # Model
        self.model = Model([encoder_inputs, decoder_inputs], decoder_outputs, name="TimeSeriesAutoencoder")
        return self.model

    def pretrain(self, train_data, val_data, epochs, batch_size):
        # Compile model
        optimizer = Adam(lr=self.learning_rate)
        self.model.compile(optimizer=optimizer, loss="mse")

        
        # train model
        self.model.fit(
            x=train_data,
            y=train_data[1:], # shifted version of the input data serves as the target
            epoch=epochs,
            batch_size=batch_size,
            validation_data=val_data,
            callbacks=[
                EarlyStopping(monitor="val_loss", patience=self.patience, restore_best_weights=True)
            ]
        )
    
    def load_weights(self, path):
        self.model.load_weights(path)
    
    def save_weights(self, path):
        self.model.save_weights(path)
    

    # train with online learning using SGD
    # train_data: numpy array of shape (n_samples, seq_len, n_features), ground truth data
    # val_data: numpy array of shape (n_samples, seq_len, n_features), ground truth data, use most recent 20% split of buffer
    # epochs: number of epochs to train, repeating the same batches
    # batch_size: number of batches to train on. Should be atleast output_seq_len+1, otherwise the model will not be able to learn anything. Round up to a power of 2 for better performance
    # output_seq_len: number of steps to predict into the future, should be atleast 1, should match output_seq_len in TimeSeriesAutoencoder
    def train_online(self, train_data, val_data, epochs, batch_size, output_seq_len):
        # Compile model
        optimizer = Adam(lr=self.learning_rate)
        self.model.compile(optimizer=optimizer, loss="mse")

        # Online training loop
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")

            # Shuffle data
            np.random.shuffle(train_data)

            # Train on batches
            n_batches = len(train_data) // batch_size
            for i in range(n_batches):
                batch_data = train_data[i * batch_size: (i + 1) * batch_size]
                x_batch = batch_data[:, :-output_seq_len, :]
                y_batch = batch_data[:, output_seq_len:, :]

                loss = self.model.train_on_batch([x_batch, x_batch], y_batch)
            
            # Evaluate on validation data
            val_loss = self.model.evaluate(val_data[:, :-1, :], val_data[:, 1:, :], verbose=0)

            print(f"Training loss: {loss:.4f} | Validation loss: {val_loss:.4f}")
    
    
    def predict(self, x_input):
        # Encode input sequence
        encoder_output = self.encoder(x_input)

        # Decode encoded sequence
        decoder_output = self.decoder(encoder_output)
        
        # Return both decoder output and encoder embedding
        return decoder_output, encoder_output



    def encode(self, data):
        return self.encoder.predict(data)
    
    def decode(self, data):
        return self.decoder.predict(data)
    

# data: 2D time series data of shape (timesteps, n_features)
# seq_len: length of sequence to split data into
# step: number of timesteps to shift the window by at each step
def sliding_window(data, seq_len, step=1):
    n_samples = (data.shape[0] - seq_len) // step + 1 # number of samples
    result = np.zeros((n_samples, seq_len, data.shape[1])) # initialize result array
    for i in range(n_samples):
        result[i] = data[i * step: i * step + seq_len]
    return result


# generator: generator function that yields 2d numpy arrays of shape (timesteps, n_features)
# model: TimeSeriesAutoencoder object
# n_predict_samples: how many samples in our buffer til we can start predicting
# buffer_train_size: how many samples in our buffer til we can start training

def largest_power_of_2_above(n):
    return 2 ** (n - 1).bit_length()

def train_and_predict(generator, model:TimeSeriesAutoencoder, n_predict_samples, n_train_samples, output_seq_len, save_weights_on_epoch=None):
    # Initialize buffer
    buffer = []

    # Create thread lock
    buffer_lock = Lock()

    # Initialize prediction queue
    training_queue = Queue()
    
    def train_thread():
        epoch_counter = 0
        while True:
            # consume prediction tasks from the queue
            training_batches_list = []
            while not training_queue.empty():
                training_batches = training_queue.get()
                training_batches_list.append(training_batches)
            
            if len(training_batches_list) > 0:
                # Concatenate prediction tasks to create training data
                train_data = np.concatenate(training_batches_list)

                # train the model online
                with buffer_lock:
                    # split data into train and validation sets
                    n_samples = len(buffer)
                    val_size = int(0.2 * n_samples) # 20% validation size
                    train_data = np.array(train_data[:-val_size])
                    val_data = np.array(train_data[-val_size:])
                    batch_size = largest_power_of_2_above(output_seq_len + 1)

                    # Train model online
                    model.train_online(train_data, val_data, epochs=1, batch_size=batch_size, output_seq_len=output_seq_len)

                    # save model weights
                    if save_weights_on_epoch is not None:
                        if epoch_counter % save_weights_on_epoch == 0:
                            model.save_weights(f"weights/epoch_{epoch_counter}.h5")
                        epoch_counter += 1

                    # remove oldest data from the buffer longer than n_train_samples
                    buffer = buffer[-n_train_samples:]
                
            # sleep for a short time to prevent busy waiting
            time.sleep(0.1)

    # start training thread
    training_thread = Thread(target=train_thread)
    training_thread.start()


    # Loop over generator, which in this case fetches data from exchange, performs preprocessing
    for i, data in enumerate(generator):
        # add data to buffer
        with buffer_lock:
            buffer.append(data)

        # once the buffer is at least n_predict_samples, make a prediction
        if len(buffer) >= n_predict_samples:
            buffer_array = np.array(buffer)
            # buffer_array is of shape (n_samples, seq_len, n_features)
            
            # make a forecast for next timestep
            x_input = buffer_array[-n_predict_samples:]
            y_pred, encoder_embedding = model.predict(x_input)

            # Yield most recent prediction and encoder embedding
            # y_pred is of shape (1, output_seq_len, n_features)
            yield y_pred, encoder_embedding

            # if buffer larger than n_train_samples, add prediction task to queue
            if len(buffer) >= n_train_samples:

                # add prediction task to queue
                training_queue.put(buffer_array)

                # remove oldest data from buffer
                with buffer_lock:
                    buffer = buffer[-n_train_samples:]
        
        # If buffer is not full, do not yield, continue collecting data
        else:
            continue

    # Wait for training thread to finish
    training_thread.join()
        
