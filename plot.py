import pandas as pd
from onlinemlpdemo.modules.online_mlp import OnlineMLP
import matplotlib; matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from matplotlib import style; style.use('dark_background')
from matplotlib.animation import FuncAnimation
from sklearn.metrics import mean_squared_error
from math import sqrt

# Degrees of freedom -- aim is to parameter sweep along these
HIST_LENGTH = 20
PRED_LENGTH = 10
PERIOD = 1
LAYERS = [100]
N_EPOCHS = 10

# Graph labels
TITLE_PRED = f'Real-Time MLP Predictions\n\
History Length={HIST_LENGTH}, \
Forecast Length={PRED_LENGTH}, \
Period={PERIOD},\n\
Layers={LAYERS}, \
Epochs={N_EPOCHS}'
TITLE_LOSS = 'Loss (rMSE)'

X_AXIS = 'Timestep'

Y_AXIS_OBS = 'Observed Acceleration'
Y_AXIS_PRED = 'Predicted Acceleration'
Y_AXIS_LOSS = 'Loss'

# Globals
loss = -1
paused = False
timestep = 0


def main():
    mlp = OnlineMLP(HIST_LENGTH, PRED_LENGTH, LAYERS)

    series = pd.read_csv('data/Ivol_Acc_Load_3S_10STD.lvm.csv')
    series = series[['Acceleration']]

    obs_series = pd.DataFrame(columns=[X_AXIS, Y_AXIS_OBS])
    pred_series = pd.DataFrame(columns=[X_AXIS, Y_AXIS_PRED])
    loss_series = pd.DataFrame(columns=[X_AXIS, Y_AXIS_LOSS])

    series_iter = series.itertuples()
    fig, axs = plt.subplots(2, 1, constrained_layout=True)
    fig.set_size_inches(6, 8)

    def animate(i):
        global loss, timestep
        if not paused:
            accel = next(series_iter)[1]

            axs[0].clear()
            axs[0].set_title(TITLE_PRED)

            axs[1].clear()
            axs[1].set_title(TITLE_LOSS)

            obs_series.loc[len(obs_series)] = [timestep, accel]
            obs_series.plot(ax=axs[0], x=X_AXIS, y=Y_AXIS_OBS, color='silver', style='--')

            pred_accel = mlp.iterate_training_step(accel, N_EPOCHS, PERIOD)
            if pred_accel is not None:
                pred_series.loc[len(pred_series)] = [timestep + PRED_LENGTH, pred_accel]
                merged_series = pd.merge_ordered(obs_series, pred_series, on=X_AXIS, how='inner')
                if not merged_series.empty:
                    loss = sqrt(mean_squared_error(merged_series[Y_AXIS_OBS].values, merged_series[Y_AXIS_PRED].values))
                    loss_series.loc[len(loss_series)] = [timestep, loss]

            if not pred_series.empty:
                pred_series.plot(ax=axs[0], x=X_AXIS, y=Y_AXIS_PRED, color='cyan')

            if not loss_series.empty:
                loss_series.plot(ax=axs[1], x=X_AXIS, y=Y_AXIS_LOSS, color='yellow')

            axs[0].set_xlim(xmin=0)
            axs[1].set_xlim(xmin=0, xmax=axs[0].get_xlim()[1])

            timestep += 1

    def pause_ani(e):
        global paused
        paused ^= True
        if paused:
            print(loss)

    fig.canvas.mpl_connect('button_press_event', pause_ani)
    ani = FuncAnimation(fig, animate, interval=50)
    plt.show()


if __name__ == '__main__':
    main()
