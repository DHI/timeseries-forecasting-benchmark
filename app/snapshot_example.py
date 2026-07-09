import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class DummyModel:
    def __init__(self, freq: str, horizon: int):
        self.freq = freq
        self.horizon = horizon

    def predict(self, start_prediction, horizon):
        pred_index = pd.date_range(start=start_prediction, periods=horizon, freq=self.freq)
        n_predicted = len(pred_index)
        predicted = np.sin(np.linspace(0, 2 * np.pi, n_predicted)) + np.random.normal(
            0, 0.5, size=n_predicted
        )  # Example prediction
        return pd.Series(predicted, index=pred_index)


freq = "1h"
horizon = 24

model = DummyModel(freq=freq, horizon=horizon)

n_test_series = 300
test_series = pd.Series(
    np.sin(np.linspace(0, 4 * np.pi, n_test_series)) + np.random.normal(0, 0.3, size=n_test_series),
    index=pd.date_range(start="2023-01-01", periods=n_test_series, freq=freq),
)  # Example actual data

# Arbitrary start times for predictions
start_prediction_1 = pd.Timestamp("2023-01-01 20:00:00")
start_prediction_2 = pd.Timestamp("2023-01-02 14:00:00")
start_prediction_3 = pd.Timestamp("2023-01-04 06:00:00")
starts = [start_prediction_1, start_prediction_2, start_prediction_3]

# ------- Plotting ----------------------------------------------

n_history = int(horizon / 4)

fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True, layout="tight")
for i, start_prediction in enumerate(starts):
    start_nock = start_prediction - n_history * pd.Timedelta(freq)
    pred_index = pd.date_range(start=start_prediction, periods=horizon, freq=freq)
    actual_index = pd.date_range(start=start_nock, end=pred_index[-1], freq=freq)

    actual = test_series.loc[actual_index]
    predicted = model.predict(start_prediction=start_prediction, horizon=horizon)

    axes[i].plot(actual.index, actual.values, label="Actual", color="black")
    axes[i].plot(predicted.index, predicted.values, label="Predicted", color="orangered")
    axes[i].axvline(x=start_prediction, color="gray", linestyle="--")
    axes[i].annotate(
        "Prediction start",
        xy=(start_prediction, 0),
        xycoords=("data", "axes fraction"),
        xytext=(-5, 6),
        textcoords="offset points",
        fontsize=10,
        color="grey",
        va="bottom",
        ha="left",
        rotation=90,
        rotation_mode="anchor",
    )
    axes[i].set_xlim(actual.index[0], actual.index[-1])
    axes[i].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    axes[i].legend(frameon=False)
axes[0].set_ylabel("Value [units]")
plt.show()
