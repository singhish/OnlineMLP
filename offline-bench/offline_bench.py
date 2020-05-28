import argparse
from statistics import mean
import pandas as pd
import numpy as np
import tensorflow as tf
from math import sqrt

# Magic values
DATASET_SIZE = 5.0  # number of seconds of total data (train + test) to use
N_INTERVALS = 10   # number of segments of dataset over which to calculate intermediate loss


def parse_args():
    parser = argparse.ArgumentParser(description='Offline MLP Benchmark')

    # Parameters -- same as online benchmark
    parser.add_argument('-l', '--history-length', type=int, default=20)
    parser.add_argument('-f', '--forecast-length', type=int, default=5)
    parser.add_argument('-u', '--units', type=int, default=[10], nargs='*')
    parser.add_argument('-e', '--epochs', type=int, default=10)
    parser.add_argument('-s', '--save-to-csv', action='store_false')

    # Specify dataset to use in data/ directory
    parser.add_argument('--s', type=int, default=1)
    parser.add_argument('--std', type=int, default=1)

    return parser.parse_args()


# Generates train/test history windows and their corresponding forecast targets from time series data
def gen_windows(time_series, history_length, forecast_length, train_length, test_length):
    assert (train_length + test_length <= 1.0)
    train_windows, train_targets, test_windows, test_targets = [], [], [], []
    for i in range(len(time_series) - history_length - forecast_length):
        target_idx = i + history_length + forecast_length
        window = time_series[i:(i + history_length)]
        target = time_series[target_idx]
        if target_idx < int(train_length * len(time_series)):
            train_windows.append(window)
            train_targets.append(target)
        elif int(train_length * len(time_series)) <= target_idx < int((train_length + test_length) * len(time_series)):
            test_windows.append(window)
            test_targets.append(target)
    return np.array(train_windows).reshape(len(train_windows), history_length), np.array(train_targets), \
           np.array(test_windows).reshape(len(test_windows), history_length), np.array(test_targets)


def main():
    args = parse_args()
    filename = f'../data/{args.s}S_{args.std}STD.csv'

    # Read in observation data from dataset
    ts = pd.read_csv(filename).query(f'Time <= {DATASET_SIZE}')[['Observation']].values
    x_label = 'Time (s)'
    y_label = 'Predicted Acceleration (Offline)'
    
    pred_df = pd.DataFrame(columns=[x_label, y_label])  # stores predictions for all intervals

    # Train/evaluate(/make predictions with) offline MLP over all intervals
    for train_length in [(mean([i, i + 1]) / N_INTERVALS) for i in range(N_INTERVALS)]:
        test_length = 0.5 / N_INTERVALS  # amount of evaluation/prediction data to use

        # Divide dataset into training/testing data
        train_windows, train_targets, test_windows, test_targets = \
            gen_windows(ts, args.history_length, args.forecast_length, train_length, test_length)

        #########################
        ### OFFLINE MLP MODEL ###
        model = tf.keras.Sequential()
        for u in args.units:
            model.add(tf.keras.layers.Dense(u, activation='relu', input_dim=args.history_length))
        model.add(tf.keras.layers.Dense(1))
        model.compile(optimizer='adam', loss='mse')
        #########################

        # Train and evaluate offline MLP on current interval's training data
        model.fit(train_windows, train_targets, epochs=args.epochs, verbose=0)
        rmse = sqrt(model.evaluate(test_windows, test_targets, verbose=0))

        # Log output
        interval_name = f'{round(train_length - test_length, 1)}_{round(train_length + test_length, 1)}'
        print(f'{args.s},{args.std},{args.history_length},{args.forecast_length},{args.units[0]},{args.epochs},'
              f'{interval_name},{rmse}')

        if args.save_to_csv:
            # Read in temporal data from dataset 
            interval_data_query = \
                f'{train_length * DATASET_SIZE} <= Time < {(train_length + test_length) * DATASET_SIZE}'
            interval_df = pd.read_csv(filename).query(interval_data_query)[['Time']]
            
            # Calculate approximate forecast time shift and apply to dataset
            forecast_time = interval_df['Time'].values[1] - interval_df['Time'].values[0]
            interval_df['Time'] = interval_df['Time'].map(lambda v, d=forecast_time, f=args.forecast_length: v + d * f)

            # Determine predictions of offline MLP for test data
            predictions = []
            for window in test_windows:
                predictions.append(model.predict(window.reshape(1, args.history_length)).item())

            # Correct size of predictions/interval_df if needed
            if len(predictions) > interval_df.shape[0]:
                interval_df = interval_df.assign(P=predictions[:interval_df.shape[0]])
            elif len(predictions) < interval_df.shape[0]:
                interval_df = interval_df.iloc[:len(predictions)].assign(P=predictions)
            else:
                interval_df = interval_df.assign(P=predictions)

            # Append interval_df to pred_df
            interval_df = interval_df.rename(columns={'Time': x_label, 'P': y_label})
            pred_df = pred_df.append(interval_df[[x_label, y_label]])

    # Save pred_df to .csv file, if -s arg is specified
    if args.save_to_csv:
        pred_df.to_csv('offline-pred.csv')


if __name__ == '__main__':
    main()
