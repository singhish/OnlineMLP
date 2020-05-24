import argparse
import tensorflow as tf
import pandas as pd
import numpy as np
from math import sqrt

# Magic values
FILE = '1S_1STD.csv'  # Dataset to use in data/ directory
DATASET_SIZE = 2.0  # number of seconds of total data (train + test) to use


def parse_args():
    parser = argparse.ArgumentParser(description='Offline MLP Benchmark')

    # Parameters -- same as online benchmark
    parser.add_argument('-l', '--history-length', type=int, default=20)
    parser.add_argument('-f', '--forecast-length', type=int, default=5)
    parser.add_argument('-u', '--units', type=int, default=[10], nargs='*')
    parser.add_argument('-e', '--epochs', type=int, default=10)
    parser.add_argument('-t', '--train-length', type=float, default=0.4)
    parser.add_argument('-s', '--save-to-csv', action='store_true')

    return parser.parse_args()


# Generates train/test history windows and their corresponding forecast targets from time series data
def gen_windows(time_series, history_length, forecast_length, train_length):
    train_windows, train_targets, test_windows, test_targets = [], [], [], []
    for i in range(len(time_series) - history_length - forecast_length):
        target_idx = i + history_length + forecast_length
        window = time_series[i:(i + history_length)]
        target = time_series[target_idx]
        if target_idx < int(train_length * len(time_series)):
            train_windows.append(window)
            train_targets.append(target)
        else:
            test_windows.append(window)
            test_targets.append(target)
    return np.array(train_windows).reshape(len(train_windows), history_length), np.array(train_targets), \
           np.array(test_windows).reshape(len(test_windows), history_length), np.array(test_targets)


def main():
    args = parse_args()

    # Read in data using pandas as a numpy array
    ts = pd.read_csv('../data/' + FILE).query(f'Time <= {DATASET_SIZE}')[['Observation']].values

    # Process data into tf.data.Dataset objects
    train_windows, train_targets, test_windows, test_targets = \
        gen_windows(ts, args.history_length, args.forecast_length, args.train_length)

    # Compile, train, evaluate MLP
    model = tf.keras.Sequential()
    for u in args.units:
        model.add(tf.keras.layers.Dense(u, activation='relu', input_dim=args.history_length))
    model.add(tf.keras.layers.Dense(1))
    model.compile(optimizer='adam', loss='mse')

    model.fit(train_windows, train_targets, epochs=args.epochs, verbose=0)

    rmse = sqrt(model.evaluate(test_windows, test_targets, verbose=0))

    # Log output
    print(f'{args.history_length},{args.forecast_length},{args.units[0]},{args.epochs},{args.train_length},{rmse}')

    # Save predictions on evaluation data to a .csv file if -s is specified
    if args.save_to_csv:
        pred_df = pd.read_csv('../data/' + FILE) \
            .query(f'Time > {args.train_length * DATASET_SIZE} and Time <= {DATASET_SIZE}')[['Time', 'Observation']]
        delta = pred_df['Time'].values[1] - pred_df['Time'].values[0]
        pred_df['Time'] = pred_df['Time'].map(lambda v, d=delta, f=args.forecast_length: v + d * f)

        predictions = []
        for window in test_windows:
            predictions.append(model.predict(window.reshape(1, args.history_length)).item())

        x_label = 'Time (s)'
        y_label = 'Predicted Acceleration (Offline)'

        pred_df = pred_df.assign(P=predictions[:len(pred_df['Time'].values)]) \
                         .rename(columns={'Time': x_label, 'P': y_label})

        pred_df[[x_label, y_label]].to_csv('offline-pred.csv')


if __name__ == '__main__':
    main()
