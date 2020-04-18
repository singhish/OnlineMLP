from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np


class OnlineMLP:

    def __init__(self, history_length, forecast_length, delay, units, epochs):
        """
        A wrapper class for Keras-based multilayer perceptrons (MLPs) to facilitate the process of training them online
        (e.g. for making live forecasts on real-time data streams).

        :param history_length: the number of past timesteps to use for making a prediction
        :param forecast_length: the number of timesteps ahead to make a prediction at
        :param delay: the gap length in timesteps between predictions
        :param units: a list consisting of the number of units for there to be at each hidden layer of the MLP
        :param epochs: the number of epochs to spend training the model during each iteration
        """
        # Enforce that delay must be less than or equal to forecast length
        if delay > forecast_length:
            raise ValueError('Delay must be less than or equal to the forecast length!')

        # Initialize MLP model
        self._model = Sequential()
        for u in units:
            self._model.add(Dense(u, activation='relu', input_dim=history_length))
        self._model.add(Dense(1))
        self._model.compile(optimizer='adam', loss='mse')

        # Initialize buffer
        self._buffer = []
        # As the weights of the MLP are updated as the data stream continues, its training batch size is 1. Hence, the
        # buffer only needs to contain enough observations for 1 training batch and 1 subsequent prediction.
        self._buffer_capacity = history_length + forecast_length

        # Save training parameters
        self._history_length = history_length
        self._forecast_length = forecast_length
        self._delay = delay
        self._epochs = epochs

    def advance_iteration(self, obs):
        """
        Progresses training of the MLP by one iteration.

        :param obs: an observation, preferably from a real-time data stream
        :return: a prediction if the internal buffer has reached capacity, otherwise a None object
        """
        # Add observation to buffer
        self._buffer.append(obs)

        if len(self._buffer) == self._buffer_capacity:
            # Train MLP using current "batch" from buffer
            train = np.reshape(np.array(self._buffer[:self._history_length]), (1, self._history_length))
            target = np.reshape(np.array(self._buffer[-1]), (1, 1))
            self._model.fit(train, target, epochs=self._epochs, verbose=0)

            # Make prediction at forecast_length
            pred = self._model.predict(np.reshape(np.array(self._buffer[self._forecast_length:]),
                                                  (1, self._history_length))).item()

            # Move buffer forward by the delay
            self._buffer = self._buffer[self._delay:]

            return pred

        return None
