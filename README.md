# OnlineMLP
![](media/demo.gif)

A wrapper class for Keras-based multilayer perceptrons (MLPs) for training them online (e.g. for making live forecasts on real-time data streams).

To install dependencies (preferably in something like a Venv or a Conda environment):

```pip install -r requirements.txt```

To perform a benchmark run using the files in the ```data``` directory: 

```python -m benchmark```

Command line options:

- `-l`/`--history-length`: The number of past timesteps to use for making a prediction. (default: 20)
- `-f`/`--forecast-length`: The number of timesteps ahead to make a prediction at. (default: 5)
- `-u`/`--units`: The number of units in the MLP\'s hidden layer. A list of integers separated by spaces can also be
provided to specify additional layers. (default: 100)
- `-e`/`--epochs`: The number of epochs to spend training the model. (default: 10)
- `-d`/`--delay`: The gap length in timesteps between predictions. (default: 1)
- `-i`/`--iterations`: The number of iterations to use on each dataset in the `benchmark_data` directory.
(default: 10000)
- `-t`/`--time-to-run`: The length of time in seconds to predict on. (default: 5.0)
- `-g`/`--graphs`: Providing this argument will show a plot after each dataset processed.

Note the data must be of `.csv` format and have two columns named `Time` and `Observation` for this script to work
properly.

---

To carry out a parameter sweep on an HPC cluster that supports `slurm` (like, for instance, `hyperion` at UofSC):

```sbatch sweep.sh```

changing any parameter bounds by directly modifying `sweep.sh` as needed. To benchmark an offline MLP for means of
comparison, run `sbatch offline-bench/sweep.sh`.

---

To see a live demo of the MLP training in real-time, checkout the `live-demo` branch.
