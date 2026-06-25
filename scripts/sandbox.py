import matplotlib.pyplot as plt
import pandas as pd

from tsforecastbench.bench.methods.base import Forecaster
from tsforecastbench.data.registry import get_case
from tsforecastbench.data.schema import Case


class STLARIMAForecaster(Forecaster):
    name = "AutoARIMA"

    def __init__(self, season_length: int = 1, max_train_steps: int = 5000):
        self.season_length = season_length
        self.max_train_steps = max_train_steps

    def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        from sklearn.preprocessing import PowerTransformer

        self._transformer = PowerTransformer().fit(df)
        df_values = self._transformer.transform(df)
        return pd.DataFrame(df_values, index=df.index, columns=df.columns).interpolate()

    def fit(self, case: Case, train: pd.DataFrame) -> None:
        from statsforecast.models import MSTL, AutoARIMA

        df = train[["unique_id", "ds", "y"]].copy()
        if len(df) > self.max_train_steps:
            df = df.iloc[-self.max_train_steps :]
        ds = pd.to_datetime(df["ds"])
        if ds.dt.tz is not None:
            ds = ds.dt.tz_convert("UTC").dt.tz_localize(None)
        df["ds"] = ds

        self._sf = MSTL(season_length=self.season_length, trend_forecaster=AutoARIMA())

        preprocessed = self._preprocess(df[["y"]])
        self._sf.fit(preprocessed["y"])

    def predict(
        self,
        case: Case,
        history: pd.DataFrame,
        horizon: int,
        future_exog: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        y_hat = self._sf.predict(h=horizon)["mean"]
        y_hat = self._transformer.inverse_transform(y_hat.reshape(-1, 1)).flatten()
        last_ds = history["ds"].max()
        future = self._future_index(case, last_ds, horizon)
        return pd.DataFrame({"ds": future, "y_hat": y_hat})


if __name__ == "__main__":
    # Parameters -----------------------------
    case_name = "demand_ltown"  # "watertemp_james"
    season_length = 24
    horizon = 72
    train_slice = slice(-3000, -1000)
    test_slice = slice(-1000, -1000 + horizon)
    # ----------------------------------------

    case = get_case(case_name)
    model = STLARIMAForecaster(season_length=season_length)
    data = pd.read_parquet(f"data/processed/{case_name}.parquet")
    train = data.iloc[train_slice]
    model.fit(case, train)
    pred = model.predict(case, train, horizon)
    plt.plot(pred["ds"], pred["y_hat"], label="predicted")
    plt.plot(data.iloc[test_slice]["ds"], data.iloc[test_slice]["y"], label="actual")
    plt.legend()
    plt.show()
