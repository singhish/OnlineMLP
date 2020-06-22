import argparse
import pandas as pd
from modules.online_mlp import OnlineMLP
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt

# Magic values
DATASET_SIZE = 5.0  # number of seconds of total data (train + test) to use
N_INTERVALS = 10  # number of segments of dataset over which to calculate intermediate loss
FORECAST_LENGTH = 1  # fix forecast length to 1 to allow for comparability with other models
DELAY = 1  # gap length in timesteps between predictions


def parse_args():
    parser = argparse.ArgumentParser(description='Online MLP Benchmark')

    # Parameters
    parser.add_argument('-i', '--input-dim', type=int, default=50,
                        help="Specifies input dimension for MLP\'s input layer and cell size for LSTM's input layer.")
    parser.add_argument('-u', '--units', type=int, default=[30], nargs='*',
                        help="List of numbers specifying, in order, number of units in each of MLP's hidden layers.")
    parser.add_argument('-e', '--epochs', type=int, default=9,
                        help='Number of epochs to train both MLP and LSTM.')

    # Specify dataset to use in data/ directory
    parser.add_argument('--s', type=int, default=1,
                        help='Specifies dataset to train on in data/ folder: {--s}S_{--std}STD.csv')
    parser.add_argument('--std', type=int, default=1,
                        help='Specifies dataset to train on in data/ folder: {--s}S_{--std}STD.csv')

    # Other options
    parser.add_argument('-s', '--save-to-csv', action='store_false',
                        help="Saves observed values of dataset and predictions for both models to .csv's if specified.")

    return parser.parse_args()


def main():
    args = parse_args()
    filename = f'data/{args.s}S_{args.std}STD.csv'

    # Dataframe column labels
    iter_label = 'Iteration'
    x_label = 'Time (s)'
    y_label_obs = 'Observed Acceleration'
    y_label_pred_mlp = 'Predicted Acceleration (Online MLP)'
    y_label_pred_lstm = 'Predicted Acceleration (Online LSTM)'
    y_label_loss = 'Loss (Online MLP)'
    y_label_loss_lstm = 'Loss (Online LSTM)'
    interval_label = 'Interval'

    # Load dataset/initialize dataframes
    df = pd.read_csv(filename).query(f'Time <= {DATASET_SIZE}')[['Time', 'Observation']]  # Original data
    obs_df = pd.DataFrame(columns=[iter_label, x_label, y_label_obs])  # Keeps track of current observations
    pred_df = pd.DataFrame(columns=[iter_label, x_label, y_label_pred_mlp])  # Keeps track of MLP's predictions
    loss_df = pd.DataFrame(columns=[iter_label, interval_label, x_label, y_label_loss])  # Stores MLP's rmse over time
    lstm_pred_df = pd.DataFrame(columns=[iter_label, x_label, y_label_pred_lstm])  # Keeps track of LSTM's predictions
    lstm_loss_df = pd.DataFrame(columns=[iter_label, interval_label, x_label, y_label_loss_lstm]) # Stores LSTM's rmses

    # Initialize online MLP model
    omlp = OnlineMLP(args.input_dim, FORECAST_LENGTH, DELAY, args.units, args.epochs)

    # Initialize vanilla LSTM model (reference model for online MLP)
    lstm = Sequential([
        LSTM(args.input_dim, input_shape=(args.input_dim, 1)),
        Dense(1)
    ])
    lstm.compile(loss='mse', optimizer='adam')

    # Start online training
    iteration = 0  # keeps track of the current training iteration
    delta = df['Time'].values[1] - df['Time'].values[0]  # approximate number of time step between observations
    n_rows = df.shape[0]  # total number of rows in dataset
    interval = 0  # stores index of current interval rmse is being calculated over
    interval_name = ''  # stores label of current interval in format '{START_OF_INTERVAL}_{END_OF_INTERVAL}'
    curr_rmse = -1  # stores current value of rmse in interval for MLP
    lstm_curr_rmse = -1  # stores current value of rmse in interval for LSTM
    lstm_buffer = []  # rolling buffer for LSTM's observations (works like online MLP's rolling buffer)

    for row in df.itertuples():
        # Get current time and acceleration values
        time = row[1]
        accel = row[2]
        obs_df.loc[len(obs_df)] = [iteration, time, accel]

        # Pandas strings for querying/labelling data for current interval
        interval_name = f'{interval / N_INTERVALS}_{(interval + 1) / N_INTERVALS}'
        interval_data_query = f'{int((interval / N_INTERVALS) * n_rows)}' \
                              f'<= {iter_label}' \
                              f'< {int(((interval + 1) / N_INTERVALS) * n_rows)}'

        # Perform prediction/loss calculations for LSTM
        # TODO: refactor this to support varying DELAY / FORECAST_LENGTH values
        lstm_buffer.append(accel)
        if len(lstm_buffer) == args.input_dim + 1:
            # Extract observation/prediction
            lstm_train = np.array(lstm_buffer[:args.input_dim]).reshape((1, args.input_dim, 1))
            lstm_target = np.array(lstm_buffer[-1]).reshape((1, 1))

            # Update LSTM with new training data
            lstm.fit(lstm_train, lstm_target, epochs=args.epochs, verbose=0)

            # Make prediction with LSTM and store it in lstm_pred_df
            lstm_pred_df.loc[len(lstm_pred_df)] = [
                int(iteration + FORECAST_LENGTH),
                time + delta * FORECAST_LENGTH,
                lstm.predict(
                    np.array(lstm_buffer[1:(args.input_dim + 1)]).reshape((1, args.input_dim, 1))
                ).reshape(1).item()
            ]

            # Calculate rmse for current interval (exact same thing is happening with online MLP below)
            lstm_synced_df = pd.merge_ordered(
                obs_df, lstm_pred_df, on=iter_label, how='inner').query(interval_data_query)
            if not lstm_synced_df.empty:
                lstm_curr_rmse = sqrt(mean_squared_error(lstm_synced_df[y_label_obs].values,
                                                         lstm_synced_df[y_label_pred_lstm].values))
                lstm_loss_df.loc[len(lstm_loss_df)] = [iteration, interval_name, time, lstm_curr_rmse]

            # Roll forward buffer by DELAY
            lstm_buffer = lstm_buffer[1:(args.input_dim + 1)]

        # Perform a training iteration on MLP
        pred_accel = omlp.advance_iteration(accel)

        # If MLP's buffer contained enough observations to make a prediction, attempt to update rmse
        if pred_accel is not None:
            pred_df.loc[len(pred_df)] = [int(iteration + FORECAST_LENGTH), time + delta * FORECAST_LENGTH, pred_accel]

            # Perform inner join on obs_df and pred_df to sync MLP's rmse values
            synced_df = pd.merge_ordered(obs_df, pred_df, on=iter_label, how='inner').query(interval_data_query)

            if not synced_df.empty:
                # Update loss_df
                curr_rmse = sqrt(mean_squared_error(synced_df[y_label_obs].values, synced_df[y_label_pred_mlp].values))
                loss_df.loc[len(loss_df)] = [iteration, interval_name, time, curr_rmse]

                # Log loss for current interval upon reaching end of interval
                if iteration >= int((interval + 1) * (n_rows / N_INTERVALS)):
                    interval += 1  # increment interval
                    print(f'{args.s},{args.std},{args.input_dim},{args.units[0]},{args.epochs},{interval_name},'
                          f'{curr_rmse},{lstm_curr_rmse}')

        # Increment iteration count
        iteration += 1

    # Log loss for final interval
    print(f'{args.s},{args.std},{args.input_dim},{args.units[0]},{args.epochs},{interval_name},'
          f'{curr_rmse},{lstm_curr_rmse}')

    # Save obs_df, pred_df, and lstm_pred_df to a merged .csv file if -s arg is specified
    if args.save_to_csv:
        pd.merge_ordered(
            obs_df,
            pd.merge_ordered(
                pred_df,
                lstm_pred_df,
                on=iter_label,
                how='outer'
            ),
            on=iter_label,
            how='outer'
        )[[x_label, y_label_obs, y_label_pred_mlp, y_label_pred_lstm]].to_csv(
            f'predictions-{args.s}S-{args.std}STD.csv')


if __name__ == '__main__':
    main()
