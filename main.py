import numpy as np
import pandas as pd
from tests.test_meta import make_meta
from  tests.spec.test_align import minutes_series, discharge_meta
from tidesurgedata.align import to_grid

def main():

    # Create a time series
    s = minutes_series("6min", 10 * 24)

    meta = discharge_meta(sampling="window_mean", window=pd.Timedelta("25h"), label="end")

    print(meta.window)
    print(meta.label)

    #out = to_grid(s, make_meta(), "1h", how="instant")

    print(s)


if __name__ == "__main__":
    main()
