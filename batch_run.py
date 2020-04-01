import argparse
import os
import pandas as pd
from modules.online_mlp import OnlineMLP
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt


def parse_args():
    parser = argparse.ArgumentParser(description='Real-time MLP Training Demo')

    # Degrees of freedom -- for parameter sweeps on HPC clusters
    parser.add_argument('-s', '--sampling-window', type=int, default=20,
                        help='The number of timesteps to use for making a prediction. (default: 20)')
    parser.add_argument('-c', '--forecast-length', type=int, default=5,
                        help='The number of timesteps ahead to make a prediction at. (default: 10)')
    parser.add_argument('-p', '--period', type=int, default=1,
                        help='The gap length in timesteps between predictions. (default: 1)')
    parser.add_argument('-l', '--layers', action='append', type=int, default=100,
                        help='The number of units in the MLP\'s hidden layer. Providing this argument more than once '
                             'will add additional layers. (default: 100)')
    parser.add_argument('-e', '--epochs', type=int, default=10,
                        help='The number of epochs to spend training the model. (default: 10)')

    # Other arguments -- useful for local testing
    parser.add_argument('-i', '--iterations', type=int, default=10000,
                        help='The number of iterations to use on each dataset in the .data directory. (default: 10000)')
    parser.add_argument('-t', '--runtime', type=float, default=5.0,
                        help='The length of time in seconds to predict on. (default: 5.0)')
    parser.add_argument('-g', '--graphs', action='store_true',
                        help='Providing this argument will show a plot after each dataset processed.')

    return parser.parse_args()


def main():
    args = parse_args()

    # Graph/axes labels
    title_pred = f'Real-Time MLP Predictions\nSampling Window={args.sampling_window}, \
Forecast Length={args.forecast_length}, Period={args.period},\nLayers={args.layers}, Epochs={args.epochs}'
    title_loss = 'Loss (rMSE)'

    step = 'Timestep'

    x_axis = 'Time (s)'

    y_axis_obs = 'Observed Acceleration'
    y_axis_pred = 'Predicted Acceleration'
    y_axis_loss = 'Loss'

    for filename in os.listdir('data'):
        series = pd.read_csv('data/' + filename)
        series = series[['X_Value', 'Acceleration']]
        obs_series = pd.DataFrame(columns=[step, x_axis, y_axis_obs])
        pred_series = pd.DataFrame(columns=[step, x_axis, y_axis_pred])
        loss_series = pd.DataFrame(columns=[step, x_axis, y_axis_loss])

        # Initialize plots
        fig, axs = plt.subplots(2, 1)
        fig.set_size_inches(6, 8)

        # Online training
        mlp = OnlineMLP(args.sampling_window, args.forecast_length, args.layers)
        loss = -1
        timestep = 0
        delta = series['X_Value'].values[1] - series['X_Value'].values[0]
        for row in series.head(args.iterations).itertuples():
            if row[1] >= args.runtime:
                break
            accel = row[2]
            obs_series.loc[len(obs_series)] = [timestep, row[1], accel]
            pred_accel = mlp.iterate_training_step(accel, args.epochs, args.period)
            if pred_accel is not None:
                pred_series.loc[len(pred_series)] = [timestep + args.forecast_length,
                                                     row[1] + delta * args.forecast_length,
                                                     pred_accel]
                merged_series = pd.merge_ordered(obs_series, pred_series, on=step, how='inner')
                if not merged_series.empty:
                    loss = sqrt(mean_squared_error(merged_series[y_axis_obs].values, merged_series[y_axis_pred].values))
                    loss_series.loc[len(loss_series)] = [timestep, row[1], loss]
            timestep += 1

        if args.graphs:
            axs[0].set_title(title_pred)
            obs_series.plot(ax=axs[0], x=x_axis, y=y_axis_obs, color='silver', style='--')
            pred_series.plot(ax=axs[0], x=x_axis, y=y_axis_pred, color='red')
            axs[0].set_xlim(xmin=0)

            axs[1].set_title(title_loss)
            loss_series.plot(ax=axs[1], x=x_axis, y=y_axis_loss, color='magenta')
            axs[1].set_xlim(xmin=0, xmax=axs[0].get_xlim()[1])
            axs[1].set_ylim(ymin=0)

            plt.show()

        print(loss)


if __name__ == '__main__':
    main()
