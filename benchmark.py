import argparse
import os
import pandas as pd
from modules.online_mlp import OnlineMLP
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt


def parse_args():
    parser = argparse.ArgumentParser(description='Real-time MLP Benchmark')

    # Degrees of freedom -- for parameter sweeps on HPC clusters
    parser.add_argument('-l', '--history-length', type=int, default=20,
                        help='The number of past timesteps to use for making a prediction. (default: 20)')
    parser.add_argument('-f', '--forecast-length', type=int, default=5,
                        help='The number of timesteps ahead to make a prediction at. (default: 5)')
    parser.add_argument('-p', '--prediction-period', type=int, default=1,
                        help='The gap length in timesteps between predictions. (default: 1)')
    parser.add_argument('-u', '--units', type=int, default=[100], nargs='*',
                        help='The number of units in the MLP\'s hidden layer. A list of integers separated by spaces '
                             'can also be provided to specify additional layers. (default: 100)')
    parser.add_argument('-e', '--epochs', type=int, default=10,
                        help='The number of epochs to spend training the model. (default: 10)')

    # Benchmark options -- useful for local testing
    parser.add_argument('-i', '--iterations', type=int, default=10000,
                        help='The number of iterations to use on each dataset in /benchmark_data. (default: 10000)')
    parser.add_argument('-t', '--time-to-run', type=float, default=0.1,
                        help='The length of time in seconds to predict on. (default: 5.0)')
    parser.add_argument('-g', '--graphs', action='store_true',
                        help='Providing this argument will show a plot after each dataset processed.')

    return parser.parse_args()


def main():
    args = parse_args()

    # Dataframe column labels
    iter_label = 'Iteration'
    x_label = 'Time (s)'
    y_label_obs = 'Observed Acceleration'
    y_label_pred = 'Predicted Acceleration'
    y_label_loss = 'Loss'

    for filename in os.listdir('benchmark_data'):
        # Load dataset/initialize dataframes
        df = pd.read_csv('benchmark_data/' + filename)[['Time', 'Observation']]  # Original time series data
        obs_df = pd.DataFrame(columns=[iter_label, x_label, y_label_obs])  # Keeps track of current observations
        pred_df = pd.DataFrame(columns=[iter_label, x_label, y_label_pred])  # Keeps track of MLP's predictions
        loss_df = pd.DataFrame(columns=[iter_label, x_label, y_label_loss])  # Stores MLP's loss over time

        # Start online training
        mlp = OnlineMLP(args.history_length, args.forecast_length, args.prediction_period, args.units, args.epochs)
        loss = -1  # Stores the current loss of the model
        iteration = 0  # Keeps track of the current training iteration
        delta = df['Time'].values[1] - df['Time'].values[0]  # The approximate time step between observations

        for row in df.head(args.iterations).itertuples():
            # Get current time and acceleration values
            time = row[1]
            if time >= args.time_to_run:
                break
            accel = row[2]
            obs_df.loc[len(obs_df)] = [iteration, time, accel]

            # Perform a training iteration on the MLP
            pred_accel = mlp.advance_iteration(accel)

            # If MLP's buffer contained enough observations to make a prediction, attempt to calculate its loss
            if pred_accel is not None:
                pred_df.loc[len(pred_df)] = [iteration + args.forecast_length,
                                             time + delta * args.forecast_length,
                                             pred_accel]
                # Perform inner join on obs_df and pred_df to sync up MLP's loss values
                merged_series = pd.merge_ordered(obs_df, pred_df, on=iter_label, how='inner')
                if not merged_series.empty:
                    loss = sqrt(mean_squared_error(merged_series[y_label_obs].values,
                                                   merged_series[y_label_pred].values))
                    loss_df.loc[len(loss_df)] = [iteration, time, loss]

            # Increment iteration count
            iteration += 1

        # Plot data if arg is provided
        if args.graphs:
            fig, axs = plt.subplots(2, 1, tight_layout=True)
            fig.set_size_inches(6, 8)

            title_pred = f'Online MLP Training Results\nHistory Length={args.history_length}, \
Forecast Length={args.forecast_length}, Prediction Period={args.prediction_period},\nUnits={args.units}, \
Epochs={args.epochs}'
            title_loss = 'Loss (rMSE)'

            axs[0].set_title(title_pred)
            obs_df.plot(ax=axs[0], x=x_label, y=y_label_obs, color='silver', style='--')
            pred_df.plot(ax=axs[0], x=x_label, y=y_label_pred, color='red')
            axs[0].set_xlim(xmin=0)

            axs[1].set_title(title_loss)
            loss_df.plot(ax=axs[1], x=x_label, y=y_label_loss, color='magenta')
            axs[1].set_xlim(xmin=0, xmax=axs[0].get_xlim()[1])
            axs[1].set_ylim(ymin=0)

            plt.show()

        print(loss)


if __name__ == '__main__':
    main()
