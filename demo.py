import argparse
import pandas as pd
from modules.online_mlp import OnlineMLP
import matplotlib; matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.metrics import mean_squared_error
from math import sqrt

# Globals
loss = -1
paused = False
timestep = 0


def parse_args():
    parser = argparse.ArgumentParser(description='Real-time MLP Live Demo')

    # Degrees of freedom
    parser.add_argument('-l', '--history-length', type=int, default=20,
                        help='The number of timesteps to use for making a prediction. (default: 20)')
    parser.add_argument('-f', '--forecast-length', type=int, default=5,
                        help='The number of timesteps ahead to make a prediction at. (default: 5)')
    parser.add_argument('-p', '--prediction-period', type=int, default=1,
                        help='The gap length in timesteps between predictions. (default: 1)')
    parser.add_argument('-u', '--units', action='append', type=int, default=100,
                        help='The number of units in the MLP\'s hidden layer. Providing this argument more than once '
                             'will add additional layers. (default: 100)')
    parser.add_argument('-e', '--epochs', type=int, default=10,
                        help='The number of epochs to spend training the model. (default: 10)')

    # Benchmark configuration
    parser.add_argument('--filename', type=str, default='benchmark_data/1S_1STD.csv',
                        help='Relative path of CSV file to use in .data directory.')

    return parser.parse_args()


def main():
    args = parse_args()

    # Graph labels
    title_pred = f'Real-Time MLP Predictions\nSampling Window={args.history_length}, \
Forecast Length={args.forecast_length}, prediction_period={args.prediction_period},\nunits={args.units}, Epochs={args.epochs}'
    title_loss = 'Loss (rMSE)'

    x_axis = 'Timestep'

    y_axis_obs = 'Observed Acceleration'
    y_axis_pred = 'Predicted Acceleration'
    y_axis_loss = 'Loss'

    # Load data/initialize dataframes
    series = pd.read_csv(args.filename)
    series = series[['Acceleration']]
    obs_series = pd.DataFrame(columns=[x_axis, y_axis_obs])
    pred_series = pd.DataFrame(columns=[x_axis, y_axis_pred])
    loss_series = pd.DataFrame(columns=[x_axis, y_axis_loss])

    # Initialize plots
    fig, axs = plt.subplots(2, 1, constrained_layout=True)
    fig.set_size_inches(6, 8)

    # Start online training
    mlp = OnlineMLP(args.history_length, args.forecast_length, args.units)
    series_iter = series.itertuples()

    def animate(i):
        global loss, timestep
        if not paused:
            accel = next(series_iter)[1]

            axs[0].clear()
            axs[0].set_title(title_pred)

            axs[1].clear()
            axs[1].set_title(title_loss)

            obs_series.loc[len(obs_series)] = [timestep, accel]
            obs_series.plot(ax=axs[0], x=x_axis, y=y_axis_obs, color='silver', style='--')

            pred_accel = mlp.iterate_training_step(accel, args.epochs, args.prediction_period)
            if pred_accel is not None:
                pred_series.loc[len(pred_series)] = [timestep + args.forecast_length, pred_accel]
                merged_series = pd.merge_ordered(obs_series, pred_series, on=x_axis, how='inner')
                if not merged_series.empty:
                    loss = sqrt(mean_squared_error(merged_series[y_axis_obs].values, merged_series[y_axis_pred].values))
                    loss_series.loc[len(loss_series)] = [timestep, loss]

            if not pred_series.empty:
                pred_series.plot(ax=axs[0], x=x_axis, y=y_axis_pred, color='red')

            if not loss_series.empty:
                loss_series.plot(ax=axs[1], x=x_axis, y=y_axis_loss, color='magenta')

            axs[0].set_xlim(xmin=0)
            axs[1].set_xlim(xmin=0, xmax=axs[0].get_xlim()[1])
            axs[1].set_ylim(ymin=0)

            timestep += 1

    # If animation is clicked, pause it and print out current loss
    def pause_ani(e):
        global paused
        paused ^= True
        if paused:
            print(loss)

    fig.canvas.mpl_connect('button_press_event', pause_ani)
    ani = FuncAnimation(fig, animate, interval=1)
    plt.show()


if __name__ == '__main__':
    main()
