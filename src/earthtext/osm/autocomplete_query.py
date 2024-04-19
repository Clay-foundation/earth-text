import pandas as pd
import numpy as np


def autocomplete_query(
    constraints: tuple[int],  # user contraints, ex: (0,0,...,1,-1)
    df: pd.DataFrame,
    n_samples: int = 1,
):
    df = df.query("split == 'train'")
    counts = np.stack(df.onehot_count)
    kept_rows_mask = np.array((True,) * df.shape[0])  # mask of kept rows
    constrained_cols_ix = np.nonzero(constraints)[0]  # indices of constrained columns
    
    for col_ix in constrained_cols_ix:
        constraint = constraints[col_ix]
        if constraint == 1:
            kept_rows_mask *= (counts[:, col_ix] > 0)
        elif constraint == -1:
            kept_rows_mask *= (counts[:, col_ix] == 0)

    print(kept_rows_mask.sum())
    if kept_rows_mask.sum() > 0:
        df = df.iloc[kept_rows_mask]  # keep rows that satisfy the constraints
        return df.sample(n=n_samples, replace=True, axis=0, random_state=42)  # sample w replacement
    else:
        return df[0:0]  # return 0 rows