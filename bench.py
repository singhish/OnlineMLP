import argparse
import pandas as pd
from modules.online_mlp import OnlineMLP
from sklearn.metrics import mean_squared_error
from math import sqrt

# Magic values
FILE = '1S_1STD.csv'  # dataset to use in data/ directory
DATASET_SIZE = 0.2  # number of seconds of total data (train + test) to use
DELAY = 1  # gap length in timesteps between predictions
N_INTERVALS = 10  # number of segments of dataset over which to calculate intermediate loss


def parse_args():
    parser = argparse.ArgumentParser(description='Online MLP Benchmark')

    # Parameters
    parser.add_argument('-l', '--history-length', type=int, default=20,
                        help='The number of past timesteps to use for making a prediction. (default: 20)')
    parser.add_argument('-f', '--forecast-length', type=int, default=5,
                        help='The number of timesteps ahead to make a prediction at. (default: 5)')
    parser.add_argument('-u', '--units', type=int, default=[10], nargs='*',
                        help='The number of units in the MLP\'s hidden layer. A list of integers separated by spaces '
                             'can also be provided to specify additional layers. (default: 10)')
    parser.add_argument('-e', '--epochs', type=int, default=10,
                        help='The number of epochs to spend training the model. (default: 10)')
    parser.add_argument('-s', '--save-to-csv', action='store_true',
                        help='Optional flag that saves the observed and predicted results to .csv files if specified.')

    return parser.parse_args()


def main():
    args = parse_args()

    # Dataframe column labels
    iter_label = 'Iteration'
    x_label = 'Time (s)'
    y_label_obs = 'Observed Acceleration'
    y_label_pred = 'Predicted Acceleration'
    y_label_loss = 'Loss'
    interval_label = 'Interval'

    # Load dataset/initialize dataframes
    df = pd.read_csv('data/' + FILE).query(f'Time <= {DATASET_SIZE}')[['Time', 'Observation']]  # Original data
    obs_df = pd.DataFrame(columns=[iter_label, x_label, y_label_obs])  # Keeps track of current observations
    pred_df = pd.DataFrame(columns=[iter_label, x_label, y_label_pred])  # Keeps track of MLP's predictions
    loss_df = pd.DataFrame(columns=[iter_label, interval_label, x_label, y_label_loss])  # Stores MLP's rmse over time

    # Initialize online MLP model
    omlp = OnlineMLP(args.history_length, args.forecast_length, DELAY, args.units, args.epochs)

    # Start online training
    iteration = 0  # keeps track of the current training iteration
    delta = df['Time'].values[1] - df['Time'].values[0]  # approximate number of time step between observations
    n_rows = df.shape[0]  # total number of rows in dataset
    interval = 0  # stores index of current interval rmse is being calculated over
    interval_name = ''  # stores label of current interval in format '{START_OF_INTERVAL}_{END_OF_INTERVAL}'
    curr_rmse = -1  # stores current value of rmse in interval

    for row in df.itertuples():
        # Get current time and acceleration values
        time = row[1]
        accel = row[2]
        obs_df.loc[len(obs_df)] = [iteration, time, accel]

        # Perform a training iteration on MLP
        pred_accel = omlp.advance_iteration(accel)

        # If MLP's buffer contained enough observations to make a prediction, attempt to update rmse
        if pred_accel is not None:
            pred_df.loc[len(pred_df)] = [iteration + args.forecast_length,
                                         time + delta * args.forecast_length,
                                         pred_accel]

            # Perform inner join on obs_df and pred_df to sync MLP's rmse values
            interval_data_query = f'{int((interval / N_INTERVALS) * n_rows)}' \
                                  f'<= {iter_label}' \
                                  f'< {int(((interval + 1) / N_INTERVALS) * n_rows)}'
            synced_df = pd.merge_ordered(obs_df, pred_df, on=iter_label, how='inner').query(interval_data_query)

            if not synced_df.empty:
                # Update loss_df
                interval_name = f'{interval / N_INTERVALS}_{(interval + 1) / N_INTERVALS}'
                curr_rmse = sqrt(mean_squared_error(synced_df[y_label_obs].values, synced_df[y_label_pred].values))
                loss_df.loc[len(loss_df)] = [iteration, interval_name, time, curr_rmse]

                # Log loss for current interval upon reaching end of interval
                if iteration >= int((interval + 1) * (n_rows / N_INTERVALS)):
                    interval += 1  # increment interval
                    print(f'{args.history_length},{args.forecast_length},{args.units[0]},{args.epochs},'
                          f'{interval_name},{curr_rmse}')

        # Increment iteration count
        iteration += 1

    # Log loss for final interval
    print(f'{args.history_length},{args.forecast_length},{args.units[0]},{args.epochs},{interval_name},{curr_rmse}')

    # Save obs_df and pred_df to .csv files, if -s arg is specified
    if args.save_to_csv:
        obs_df.to_csv('obs.csv')
        pred_df.to_csv('pred.csv')


if __name__ == '__main__':
    main()
