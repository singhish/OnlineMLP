import argparse
import tensorflow as tf
import pandas as pd
import numpy as np
from math import sqrt

# Magic values
FILE = '1S_1STD.csv'  # Dataset to use in data/ directory
DATASET_SIZE = 1.0  # number of seconds of total data (train + test) to use


def parse_args():
    parser = argparse.ArgumentParser(description='Offline MLP Benchmark')

    # Parameters -- same as online benchmark
    parser.add_argument('-l', '--history-length', type=int, default=20)
    parser.add_argument('-f', '--forecast-length', type=int, default=5)
    parser.add_argument('-u', '--units', type=int, default=[10], nargs='*')
    parser.add_argument('-e', '--epochs', type=int, default=10)
    parser.add_argument('-t', '--train-length', type=float, default=0.4)

    return parser.parse_args()


# Generates train/test history windows and their corresponding forecast targets from time series data
def gen_windows(time_series, history_length, forecast_length, train_length, test_length=0.8):
    train_windows, train_targets, test_windows, test_targets = [], [], [], []
    for i in range(len(time_series) - history_length - forecast_length):
        window = time_series[i:(i + history_length)]
        target = time_series[i + history_length + forecast_length]
        if i < int(train_length * len(time_series)):
            train_windows.append(window)
            train_targets.append(target)
        elif i >= int(test_length * len(time_series)):
            test_windows.append(window)
            test_targets.append(target)
    return (np.array(train_windows), np.array(train_targets)), (np.array(test_windows), np.array(test_targets))


def main():
    args = parse_args()

    # Read in data using pandas as a numpy array
    ts = pd.read_csv('../data/' + FILE).query(f'Time <= {DATASET_SIZE}')[['Observation']].values

    # Process data into tf.data.Dataset objects
    train, test = gen_windows(ts, args.history_length, args.forecast_length, args.train_length)
    train_ds = tf.data.Dataset.from_tensor_slices(train)
    test_ds = tf.data.Dataset.from_tensor_slices(test)

    # Compile, train, evaluate MLP
    model = tf.keras.Sequential()
    for u in args.units:
        model.add(tf.keras.layers.Dense(u, activation='relu'))
    model.add(tf.keras.layers.Dense(1))
    model.compile(optimizer='adam', loss='mse')

    model.fit(train_ds, epochs=args.epochs, verbose=0)

    rmse = sqrt(model.evaluate(test_ds, verbose=0))

    # Log output
    print(f'{args.history_length},{args.forecast_length},{args.units},{args.epochs},{args.train_length},{rmse}')


if __name__ == '__main__':
    main()
