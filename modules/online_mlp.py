from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from collections import Iterable


class OnlineMLP:

    def __init__(self, sampling_window, forecast_length, layers):
        """
        A wrapper class for Keras-based multilayer perceptrons (MLPs) to facilitate the process of training them online
        (e.g. for making live forecasts on real-time data streams).

        :param sampling_window: the number of past timesteps to use for making a prediction
        :param forecast_length: the number of timesteps ahead to make a prediction at
        :param layers: a list consisting of the number of units to be at each hidden layer of the MLP
        """
        # Initialize MLP model
        self._model = Sequential()
        if isinstance(layers, Iterable):
            for units in layers:
                self._model.add(Dense(units, activation='relu', input_dim=sampling_window))
        else:
            self._model.add(Dense(layers, activation='relu', input_dim=sampling_window))
        self._model.add(Dense(1))
        self._model.compile(optimizer='adam', loss='mse')

        # Initialize buffer
        self._buffer = []
        # As the MLP's weights are updated as the data stream continues, its training batch size is 1. Hence the buffer
        # only needs to contain enough observations to reflect 1 batch.
        self._buffer_capacity = sampling_window + forecast_length

        # Other class constants
        self._sampling_window = sampling_window
        self._forecast_length = forecast_length

    def iterate_training_step(self, obs, n_epochs, period):
        """
        Progresses training of the MLP by one iteration.

        :param obs: an observation, preferably from a real-time data stream
        :param n_epochs: the number of epochs to spend training the model
        :param period: the gap length in timesteps between predictions
        :return: a prediction if the internal buffer has reached capacity, otherwise a None object
        """
        if period > self._forecast_length:
            raise ValueError('Period must be greater than the prediction length!')

        # Add observation to buffer
        self._buffer.append(obs)

        if len(self._buffer) == self._buffer_capacity:
            # Train MLP using current "batch" from buffer
            train = np.reshape(np.array(self._buffer[:self._sampling_window]), (1, self._sampling_window))
            target = np.reshape(np.array(self._buffer[-1]), (1, 1))
            self._model.fit(train, target, epochs=n_epochs, verbose=0)

            # Make prediction at pred_length
            pred = self._model.predict(np.reshape(np.array(self._buffer[self._forecast_length:]),
                                                  (1, self._sampling_window))).item()

            # Move buffer forward by the period
            self._buffer = self._buffer[period:]

            return pred

        return None
