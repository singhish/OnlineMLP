# OnlineMLP
A wrapper class for Keras-based multilayer perceptrons (MLPs) to facilitate the process of training them online (e.g.
for making live forecasts on real-time data streams).

To install dependencies (preferably in something like a Venv or a Conda environment):

```pip install -r requirements.txt```

To run:

```python -m demo```

Command line flags:
- `-f`/`--filename`: Relative path of CSV file to use in .data directory
- `-s`/`--hist-length`: The number of past timesteps to use for making a prediction
- `-p`/`--pred-length`: The number of timesteps ahead to make a prediction at
- `-t`/`--period`: The gap length in timesteps between predictions
- `-l`/`--layers`: The number of units in the MLP's hidden layer (providing this argument more than once will add
                   additional layers)
- `-e`/`--epochs`: The number of epochs to spend training the model

\*Note: clicking on the live plot pauses it and reports the model's current rMSE loss.