"""This module contains code to extract stratified samples from datasets."""

import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def filter_single_subgroups(data: pd.DataFrame, strata: List[str]) -> pd.DataFrame:
    """Filter out rows that have a single observation in the stratas."""

    # Perform a value count over the stratas
    val_counts = data[strata].value_counts()

    # Create a dataframe with stratas that have only one observation
    single_stratas = val_counts[val_counts == 1].reset_index()

    if single_stratas.empty:
        # If no single stratas are found, return the original data
        return data

    # Pop the count column to be able to iterate over the columns of interest to
    # create the filter
    single_stratas.pop("count")

    logical_and_list = []
    for _, row in single_stratas.iterrows():
        # Create filter that accounts for data point that are not in the strata and append
        # it to the list of filters to be applied
        not_banned_observations = ~np.logical_and.reduce(
            [data[col] == value for col, value in row.items()]
        )
        logical_and_list.append(not_banned_observations)

    # Bound all the filters with a logical and
    logical_and_filter = np.logical_and.reduce(logical_and_list)

    return data[logical_and_filter]


def split_data(
    data: pd.DataFrame,
    train_size: Optional[int] = 0.8,
    strata: Optional[List[str]] = [],
    random_state: Optional[int] = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into train, validation and test sets."""

    # Filter out single stratas
    filtered_data = filter_single_subgroups(data, strata) if strata else data

    # Compute number of strata
    n_strata = filtered_data[strata].value_counts().shape[0]

    # Compute size of test set
    n_test = int(np.floor(len(filtered_data) * (1 - train_size)))

    if n_test < n_strata:
        logging.info(
            "Not enough data to create a test set with at least one observation per strata."
            " Will recompute train_size so that test set has same number of observations"
            " as the number of stratas."
        )
        train_size = 1 - n_strata / len(filtered_data)

    # Split data into train and test
    train, test = train_test_split(
        filtered_data,
        train_size=train_size,
        random_state=random_state,
        stratify=filtered_data[strata] if strata else None,
    )

    return train, test


def stratified_sampling(
    data: pd.DataFrame,
    desired_proportion: float,
    strata: Optional[List[str]] = [],
    random_state: Optional[int] = 42,
) -> pd.DataFrame:
    """Sample data to obtain a desired number of defects and a desired proportion of defects."""

    # Compute number of instances per class
    val_counts = data.Defect.value_counts()
    n_defects = val_counts[1]
    n_non_defects = val_counts[0]

    # Compute max number of defects based on desired proportion
    max_desired_defects = int(
        n_non_defects * desired_proportion / (1 - desired_proportion)
    )
    desired_defects = min(max_desired_defects, n_defects)

    # Compute number of desired non-defects to sample
    desired_non_defects = int(
        desired_defects * (1 - desired_proportion) / desired_proportion
    )

    # Set split proportion to sample desired number of defects
    proportion_to_obtain_defects = desired_defects / n_defects

    # Perform stratified split to obtain desired number of defects
    our_dataset, remainder = split_data(
        data,
        train_size=proportion_to_obtain_defects,
        strata=strata,
        random_state=random_state,
    )

    # Compute number of missing desired non-defects and remaining non-defects
    sampled_non_defects = our_dataset.Defect.value_counts()[0]
    missing_non_defects = desired_non_defects - sampled_non_defects
    remaining_non_defects = remainder.Defect.value_counts()[0]

    if missing_non_defects <= 0:
        logging.info(
            "Enough non-defects to sample from first split. Will downsample to obtain"
            " the desired proportion. Consider increasing the number of defects to sample"
            " to obtain a bigger sample with same proportion."
        )

        # Set split proportion to sample desired number of non-defects
        proportion_to_obtain_non_defects = desired_non_defects / sampled_non_defects

        # Down sample non-defects
        keep_sample, filter_sample = split_data(
            our_dataset,
            train_size=proportion_to_obtain_non_defects,
            strata=strata,
            random_state=random_state,
        )

        # Concatenate the defects of the sample to filter to the sample to keep
        our_dataset = pd.concat([keep_sample, filter_sample[filter_sample.Defect == 1]])

    elif missing_non_defects > remaining_non_defects:
        logging.info(
            "Not enough non-defects to sample, will use the whole sample of remaining non-defects"
        )

        # Concatenate the remainder to the dataset
        our_dataset = pd.concat([our_dataset, remainder[remainder.Defect == 0]])

    else:
        # Set split proportion to sample desired number of non-defects
        proportion_to_obtain_non_defects = missing_non_defects / remaining_non_defects

        # Sample non-defects
        remainder_sample, _ = split_data(
            remainder,
            train_size=proportion_to_obtain_non_defects,
            strata=strata,
            random_state=random_state,
        )

        # Concatenate the non-defects of the remainder to the dataset
        our_dataset = pd.concat(
            [our_dataset, remainder_sample[remainder_sample.Defect == 0]]
        )

    return our_dataset
