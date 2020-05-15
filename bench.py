import argparse
import pandas as pd
from modules.online_mlp import OnlineMLP
from sklearn.metrics import mean_squared_error
from math import sqrt

# Magic values
FILE = '1S_1STD.csv'  # Dataset to use in data/ directory
DATASET_SIZE = 2.0  # number of seconds of total data (train + test) to use
DELAY = 1  # gap length in timesteps between predictions
N_RMSES = 5  # total number of rmse measurements to report for plotting, evenly spaced across dataset


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

    return parser.parse_args()


def main():
    args = parse_args()

    # Dataframe column labels
    iter_label = 'Iteration'
    x_label = 'Time (s)'
    y_label_obs = 'Observed Acceleration'
    y_label_pred = 'Predicted Acceleration'
    y_label_loss = 'Loss'

    # Load dataset/initialize dataframes
    df = pd.read_csv('data/' + FILE).query(f'Time <= {DATASET_SIZE}')[['Time', 'Observation']]  # Original data
    obs_df = pd.DataFrame(columns=[iter_label, x_label, y_label_obs])  # Keeps track of current observations
    pred_df = pd.DataFrame(columns=[iter_label, x_label, y_label_pred])  # Keeps track of MLP's predictions
    loss_df = pd.DataFrame(columns=[iter_label, x_label, y_label_loss])  # Stores MLP's rmse over time

    # Start online training
    omlp = OnlineMLP(args.history_length, args.forecast_length, DELAY, args.units, args.epochs)
    curr_rmse = -1  # Stores the current rmse of the model
    iteration = 0  # Keeps track of the current training iteration
    delta = df['Time'].values[1] - df['Time'].values[0]  # The approximate time step between observations
    n_rows = df.shape[0]  # Total number of rows in dataset
    rmses = []
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
            synced_df = pd.merge_ordered(obs_df, pred_df, on=iter_label, how='inner')
            if not synced_df.empty:
                curr_rmse = sqrt(mean_squared_error(synced_df[y_label_obs].values,
                                                    synced_df[y_label_pred].values))
                loss_df.loc[len(loss_df)] = [iteration, time, curr_rmse]

        # If current iteration's rmse is to be reported, save it for logging
        if iteration + 1 in [int(n_rows*((i+1)/N_RMSES)) for i in range(N_RMSES)]:
            rmses.append(curr_rmse)

        # Increment iteration count
        iteration += 1

    # Log output
    print(f'{args.history_length},{args.forecast_length},{args.units[0]},{args.epochs},{",".join(map(str, rmses))}')


if __name__ == '__main__':
    main()
