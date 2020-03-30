# OnlineMLP
A wrapper class for Keras-based multilayer perceptrons (MLPs) to facilitate the process of training them online (e.g.
for making live forecasts on real-time data streams).

To install dependencies (preferably in something like a Venv or a Conda environment):

```pip install -r requirements.txt```

---

To run an animated, real-time demonstration of the MLP:

```python -m demo```

Command line flags:
- `-f`/`--filename`: Relative path of CSV file to use in .data directory
- `-s`/`--sampling-window`: The number of timesteps to use for making a prediction
- `-c`/`--forecast-length`: The number of timesteps ahead to make a prediction at
- `-t`/`--period`: The gap length in timesteps between predictions
- `-l`/`--layers`: The number of units in the MLP's hidden layer (providing this argument more than once will add
                   additional layers)
- `-e`/`--epochs`: The number of epochs to spend training the model

\*Note: clicking on the live plot pauses it and reports the model's current rMSE loss.

---


To perform a batch run on each file in the ```data``` directory: 

```python -m batch_run```

Command line flags are the same as above, with the exception of the `-f`/`--filename` option being replaced with an
`-i`/`--iterations` option.