import kagglehub
import pandas as pd
import numpy as np
from sklearn.preprocessing import (
    LabelEncoder,
)  # to encode categorical features as integers
import torch
from torch.utils.data import TensorDataset
import os  # to save later
import time
import shutil


def download_dataset(
    year_start,
    year_end_exd,
    origin_path="flnny123/mfddmulti-modal-flight-delay-dataset/versions/4",
    mode="tabular",  # or "sequential" for pre-made chains
    output_dir_seq="data/chain/",
):

    dest_paths = []
    for year in range(year_start, year_end_exd):
        print(f"Downloading year {year} data...")
        if mode == "tabular":
            origin_path_year = (
                "Aeolus/Flight_Tab/flight_with_weather_" + str(year) + ".csv"
            )
            dest_path_year = kagglehub.dataset_download(
                origin_path, path=origin_path_year
            )
            dest_paths.append(dest_path_year)

        # dest_path_year = dest_path+'flight_with_weather_'+str(year)+'.csv' # destination path
        elif mode == "sequential":
            for split in ["train", "val", "test"]:
                origin_path_year_split = f"Aeolus/Flight_chain/chain_data_{year}/{split}_flight_chain_{year}.pt"
                final_path = os.path.join(
                                    output_dir_seq + str(year), f"{split}_flight_chain_{year}.pt"
                                )
                if os.path.exists(final_path): 
                    print(f"Path {final_path} already exists! Skipping it")
                    continue
                dest_path_year = kagglehub.dataset_download(
                    origin_path, path=origin_path_year_split
                )
                os.makedirs(output_dir_seq + str(year), exist_ok=True)
                shutil.move(dest_path_year, final_path)
                dest_path_year = final_path
                dest_paths.append(final_path)

                print(f"(File(s) available at {dest_path_year}).")

    return dest_paths


def read_dataset_pandas(
    file_path_year, exploring=True
):  # if True, get limited number of rows to avoid memory issues):
    print("Converting to Pandas dataframe...")
    if exploring:
        df = pd.read_csv(file_path_year, nrows=10000)
    else:
        df = pd.read_csv(file_path_year)
    print("All done.")
    return df

def load_dataset_pytorch(year, file_path= "data/chain/"): # for sequential when directly available
    split_types = ["train", "val", "test"] 

    loaded_data = {}

    for split in split_types:
        full_file_path = file_path+ f"{year}/{split}_flight_chain_{year}.pt"
        loaded_data[split] = torch.load(full_file_path, weights_only=False)

        print(f"--- Read file: (split: {split}, year: {year}) ---")

    return loaded_data

def remove_outliers_percentile(df, columns, lower=0.01, upper=0.99):  # good practice
    n_rows_before = len(df)
    for col in columns:
        lower_bound = df[col].quantile(lower)
        upper_bound = df[col].quantile(upper)
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    n_rows_after = len(df)
    percentage_dropped = 100 * (n_rows_before - n_rows_after) / n_rows_before
    print(f"Dropped {percentage_dropped:.2f} % of rows because outliers.")

    return df


def clean_dataframe(
    df,
    int_type="int32",
    float_type="float32",
    cache_bool=False,
):  # cache=False to save memory at the cost of speed
    # type conversions
    for col in DATE_COLS + DATETIME_COLS:
        df[col] = pd.to_datetime(df[col], format="%Y-%m-%d %H:%M:%S", cache=cache_bool)
    for col in DATE_COLS:
        df[col] = df[col].dt.normalize()  # keep date only, no time (cleaner)
    for col in TIMEDELTA_MINS_COLS:
        df[col] = pd.to_timedelta(df[col], unit="m")
    df[INT_COLS] = df[INT_COLS].astype(int_type)
    df[STR_COLS] = df[STR_COLS].astype("str")
    df[FLOAT_COLS] = df[FLOAT_COLS].astype(float_type)  # limit precision??

    # drop all rows containing NaNs and keep track of them
    n_rows_before = len(df)
    df.dropna(subset=["D_TEMP", "D_PRCP", "D_WSPD"], inplace=True)
    n_rows_after = len(df)
    n_rows_dropped = n_rows_before - n_rows_after
    print(f"Dropped {n_rows_dropped} rows because of NaNs.")
    nan_bool = bool(np.any(df_2022.isna().sum() > 0))
    print(f"Any NaNs remaining in numerical data: {nan_bool}.")

    return df


def create_flight_chains(df):
    df_sorted = df.sort_values(
        by=["OP_CARRIER", "OP_CARRIER_FL_NUM", "FL_DATE", "CRS_DEP_TIME"]
    )
    grouped = df_sorted.groupby(
        ["OP_CARRIER", "OP_CARRIER_FL_NUM", "FL_DATE"]
    )  # unique aircraft identifiers
    res = {name: group for name, group in grouped}  # dict
    return res


# Truncation/padding for single chain
def adjust_sequence(data, max_len):
    if len(data) < max_len:
        pad_shape = (max_len - len(data), data.shape[1])
        return torch.cat([data, torch.zeros(pad_shape, dtype=data.dtype)], dim=0)
    return data[:max_len]


def process_all_chains(
    flight_chains,
    target_columns,
    dense_feat_cols,
    sparse_feat_cols,
    max_sequence_length,
    year,  # for saving to torch dataset with proper name
):
    processed = []
    for name, chain in flight_chains.items():
        dense_tensors = [
            torch.tensor(
                chain[name].fillna(0).values, dtype=torch.float32
            )  # handle NaNs; NNs expect float
            for name in dense_feat_cols
        ]
        dense_feat = torch.stack(
            dense_tensors, dim=1
        )  # stacks columns horizontally, resulting shape (seq_len, num_dense_features)

        # feature engineering
        chain["MONTH"] = (chain["MONTH"] - 1).astype(
            np.int16
        )  # standard 0-based embedding + store as int16 to save memory
        chain["DAY_OF_WEEK"] = (chain["DAY_OF_WEEK"] - 1).astype(np.int16)

        sparse_tensors = [
            torch.tensor(
                chain[name].values.astype(np.int16), dtype=torch.int16
            )  # categorical columns should not have NAs
            for name in sparse_feat_cols
        ]
        sparse_feat = torch.stack(sparse_tensors, dim=1)

        # labels for binary classification + full delays
        # have to convert to minutes since it's a timedelta object
        delays = torch.tensor(
            (
                chain[target_columns].apply(lambda s: s.dt.total_seconds() / 60)
            ).values.astype(np.int16),
            dtype=torch.int16,
        )

        labels = torch.tensor(
            np.column_stack(
                (
                    (chain["ARR_DELAY"].dt.total_seconds() / 60 > 15).astype(
                        np.int8
                    ),  # definition of delay if more than 15 min (for binary classification)
                    (chain["ARR_DELAY"].dt.total_seconds() / 60 > 15).astype(np.int8),
                )
            ),
            dtype=torch.int8,
        )

        # Sequence length processing
        valid_len = min(
            len(dense_feat), max_sequence_length
        )  # stores the actual length before padding (but after truncation). Useful later eg in loss calculation
        dense_feat = adjust_sequence(dense_feat, max_sequence_length)
        sparse_feat = adjust_sequence(sparse_feat, max_sequence_length)
        labels = adjust_sequence(labels, max_sequence_length)
        delays = adjust_sequence(delays, max_sequence_length)

        processed.append((dense_feat, sparse_feat, labels, valid_len, delays))
    return processed


def create_dataset(processed_data):
    dense = torch.stack([item[0] for item in processed_data])
    sparse = torch.stack([item[1] for item in processed_data])
    labels = torch.stack([item[2] for item in processed_data])
    valid_lens = torch.tensor([item[3] for item in processed_data], dtype=torch.long)
    delays = torch.stack([item[4] for item in processed_data])
    return TensorDataset(dense, sparse, labels, valid_lens, delays)


# IMPORTANT NOTE: this modifies dataframe in-place;
# if mistaken, go back to read_dataset_pandas(...)
def prepare_data(
    df,
    year,
    mode="tabular",  # tabular or sequential
    target_columns=["DEP_DELAY", "ARR_DELAY"],  # in minutes
    time_of_prediction="departure",
    seed=42,
    max_sequence_length=6,
    train_frac=0.6,
    valid_frac=0.2,
):  # departure or arrival (for tabular mode)

    print(f"Start processing data for {year}...")
    start_time = time.time()
    if mode == "tabular":
        df = remove_outliers_percentile(df, target_columns)
        df = df.dropna(subset=["FL_DATE"])
        df["FL_YEAR"] = df["FL_DATE"].dt.year  # flight year...
        df.drop(columns=["FL_DATE"], inplace=True)  # ... instead of full date

        time_columns = [
            "CRS_DEP_TIME",
            "DEP_TIME",
            "WHEELS_OFF",
            "WHEELS_ON",
            "CRS_ARR_TIME",
            "ARR_TIME",
        ]
        for col in time_columns:
            df[col + "_MIN"] = df[col].dt.hour * 60 + df[col].dt.minute
            # convert time columns in minutes since midnight...
        df.drop(columns=time_columns, inplace=True)  # ... and drop the originals

        categorical_columns = [
            "OP_CARRIER",
            "OP_CARRIER_FL_NUM",
            "FL_YEAR",
            "MONTH",
            "DAY_OF_MONTH",  # unique identifier for any day;
            # scheduled?? if not, minor leak
            "ORIGIN",
            "DEST",
        ]

        # info available before actual departure time
        continuous_columns = [
            "CRS_DEP_TIME_MIN",
            "CRS_ARR_TIME_MIN",
            "FLIGHTS",
            "O_TEMP",
            "O_PRCP",
            "O_WSPD",
            "D_TEMP",
            "D_PRCP",
            "D_WSPD",
            "O_LATITUDE",
            "O_LONGITUDE",
            "D_LATITUDE",
            "D_LONGITUDE",
        ]
        if time_of_prediction == "arrival":
            # additional info available before actual arrival time (but after actual departure)
            continuous_columns = continuous_columns + [
                "DEP_TIME_MIN",
                "WHEELS_OFF_MIN",  # could also try adding "TAXI_IN"?
            ]
            target_cols = ["ARR_DELAY"]
        else:
            target_cols = target_columns
        df = df[target_cols + categorical_columns + continuous_columns]
        return df

    elif mode == "sequential":
        # feature engineering
        df["CRS_DEP_TIME_HOUR"] = df["CRS_DEP_TIME"].dt.hour.astype("int8")
        df["CRS_ARR_TIME_HOUR"] = df["CRS_ARR_TIME"].dt.hour.astype("int8")

        # encode categorical features
        encoder = LabelEncoder()
        df["OP_CARRIER"] = encoder.fit_transform(df["OP_CARRIER"])
        df["OP_CARRIER_FL_NUM"] = encoder.fit_transform(df["OP_CARRIER_FL_NUM"])

        # more feature engineering
        df["MONTH"] = df["FL_DATE"].dt.month
        df["DAY_OF_YEAR"] = df["FL_DATE"].dt.dayofyear

        dense_feat_cols = [  # numeric, all stored as floats;
            # relationship between them is quantitative (eg it makes sense to think that twice the flights will somehow have twice the impact on delay)
            "O_TEMP",
            "D_TEMP",  # assuming measured at departure??
            "O_PRCP",
            "D_PRCP",
            "O_WSPD",
            "D_WSPD",
            "FLIGHTS",
        ]

        sparse_feat_cols = [  # categorical, stored as ints
            "MONTH",
            "DAY_OF_WEEK",
            "CRS_ARR_TIME_HOUR",
            "CRS_DEP_TIME_HOUR",
            "ORIGIN_INDEX",
            "DEST_INDEX",
            "OP_CARRIER",
            "OP_CARRIER_FL_NUM",
        ]

        target_cols = target_columns

        # df = df[target_cols + ["DAY_OF_YEAR"] + ["FL_DATE"] + dense_feat_cols + sparse_feat_cols]
        # from here on we include train/test split (without leaking), torch implementation, etc.
        # (group by to create multiple chains each identified by starting day of year, etc)
        # and also time of prediction

        df = df.sort_values(by="FL_DATE").reset_index(drop=True)
        date_dim = (
            df[["FL_DATE"]].drop_duplicates()
        )  # selects FL_DATE column as a dataframe to get only unique dates
        # new columsna for unique identifier
        date_dim["DAY_OF_YEAR"] = date_dim["FL_DATE"].dt.dayofyear
        date_dim["MONTH"] = date_dim["FL_DATE"].dt.month

        rng = np.random.RandomState(seed=seed)

        train_days, valid_days, test_days = [], [], []
        for month in sorted(date_dim["MONTH"].unique()):  # 1, 2, 3, ...
            month_dates = date_dim[date_dim["MONTH"] == month]["DAY_OF_YEAR"]
            n_days = len(month_dates)
            if n_days < 3:
                raise Exception(
                    f"Allocated days less than three for {month} with this seed."
                )
            #    alloc_days = list(month_dates) * 3
            #    mandatory_days = alloc_days[:3]
            # else:
            #
            mandatory_days = rng.choice(month_dates, 3, replace=False)

            train_days.append(mandatory_days[0])
            valid_days.append(mandatory_days[1])
            test_days.append(mandatory_days[2])

            remaining_days = [d for d in month_dates if d not in mandatory_days]
            n_remaining = len(remaining_days)

            if n_remaining > 0:
                permuted = rng.permutation(remaining_days)
                split1 = int(round(n_remaining * train_frac))
                split2 = split1 + int(round(n_remaining * valid_frac))

                train_days.extend(permuted[:split1])
                valid_days.extend(permuted[split1:split2])
                test_days.extend(permuted[split2:])

        date_dim["SPLIT_TYPE"] = np.select(
            [
                date_dim["DAY_OF_YEAR"].isin(train_days),
                date_dim["DAY_OF_YEAR"].isin(valid_days),
                date_dim["DAY_OF_YEAR"].isin(test_days),
            ],
            ["train", "valid", "test"],
            default="undefined",  # in case something goes wrong, split type will not be automatically deduced
        )

        df = pd.merge(df, date_dim[["FL_DATE", "SPLIT_TYPE"]], on="FL_DATE", how="left")

        train_chains = create_flight_chains(df[df["SPLIT_TYPE"] == "train"])
        valid_chains = create_flight_chains(df[df["SPLIT_TYPE"] == "valid"])
        test_chains = create_flight_chains(df[df["SPLIT_TYPE"] == "test"])

        train_processed = process_all_chains(
            train_chains,
            target_columns=target_columns,
            dense_feat_cols=dense_feat_cols,
            sparse_feat_cols=sparse_feat_cols,
            max_sequence_length=max_sequence_length,
            year=year,
        )
        valid_processed = process_all_chains(
            valid_chains,
            target_columns=target_columns,
            dense_feat_cols=dense_feat_cols,
            sparse_feat_cols=sparse_feat_cols,
            max_sequence_length=max_sequence_length,
            year=year,
        )
        test_processed = process_all_chains(
            test_chains,
            target_columns=target_columns,
            dense_feat_cols=dense_feat_cols,
            sparse_feat_cols=sparse_feat_cols,
            max_sequence_length=max_sequence_length,
            year=year,
        )

        train_dataset = create_dataset(train_processed)
        valid_dataset = create_dataset(valid_processed)
        test_dataset = create_dataset(test_processed)

        # 5. Save results
        output_dir = f"processed_data_{year}"
        os.makedirs(output_dir, exist_ok=True)

        torch.save(
            train_dataset, os.path.join(output_dir, f"train_flight_chain_{year}.pt")
        )
        torch.save(
            valid_dataset, os.path.join(output_dir, f"val_flight_chain_{year}.pt")
        )
        torch.save(
            test_dataset, os.path.join(output_dir, f"test_flight_chain_{year}.pt")
        )

        elapsed = time.time() - start_time
        print(
            f"Finished processing data for {year}, time elapsed: {elapsed:.2f} seconds"
        )
        return f"Torch dataset available at relative path: {output_dir}"
