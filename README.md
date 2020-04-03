# OnlineMLP
A wrapper class for Keras-based multilayer perceptrons (MLPs) to facilitate the process of training them online (e.g.
for making live forecasts on real-time data streams).

To install dependencies (preferably in something like a Venv or a Conda environment):

```pip install -r requirements.txt```

---

To perform a benchmark run using the files in the ```data``` directory, run: 

```python -m benchmark```

Note the data must be of `.csv` format and have two columns named `Time` and `Observation` for this script to work
properly.

Command line options:

- `-l`/`--history-length`: The number of past timesteps to use for making a prediction. (default: 20)
- `-f`/`--forecast-length`: The number of timesteps ahead to make a prediction at. (default: 5)
- `-p`/`--prediction-period`: The gap length in timesteps between predictions. (default: 1)
- `-u`/`--units`: The number of units in the MLP\'s hidden layer. A list of integers separated by spaces can also be
provided to specify additional layers. (default: 100)
- `-e`/`--epochs`: The number of epochs to spend training the model. (default: 10)
- `-i`/`--iterations`: The number of iterations to use on each dataset in the `benchmark_data` directory.
(default: 10000)
- `-t`/`--time-to-run`: The length of time in seconds to predict on. (default: 5.0)
- `-g`/`--graphs`: Providing this argument will show a plot after each dataset processed.

---

To run a live demo of the MLP training in real-time, checkout the `live-demo` branch.
