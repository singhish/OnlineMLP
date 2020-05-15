# OnlineMLP
![](media/demo.gif)

A wrapper class for Keras-based multilayer perceptrons (MLPs) for training them online (e.g. for making live forecasts on real-time data streams).

To install dependencies (preferably in something like a `venv` or `conda` environment):

```pip install -r requirements.txt```

To perform a benchmark, run: 

```python -m bench```

Command line options:

- `-l`/`--history-length`: The number of past timesteps to use for making a prediction. (default: 20)
- `-f`/`--forecast-length`: The number of timesteps ahead to make a prediction at. (default: 5)
- `-u`/`--units`: The number of units in the MLP\'s hidden layer. A list of integers separated by spaces can also be
provided to specify additional layers. (default: 10)
- `-e`/`--epochs`: The number of epochs to spend training the model. (default: 10)

Note the data must be of `.csv` format and have two columns named `Time` and `Observation` for this script to work
properly.

---

To carry out a parameter sweep on an HPC cluster that supports `slurm` (like, for instance, `hyperion` at UofSC):

```sbatch sweep.sh```

changing any parameter bounds by directly modifying `sweep.sh` as needed. To benchmark an offline MLP for means of
comparison, run `sbatch offline-bench/sweep.sh`.

---

To see a live demo of the MLP training in real-time, checkout the `live-demo` branch.
