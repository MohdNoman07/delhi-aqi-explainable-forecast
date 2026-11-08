"""
tests/test_split.py

Unit tests for chronological splitting:
1. No overlap between Train, Validation, and Test sets.
2. Every date in an earlier set is strictly before every date in a later set.
3. The gap between sets is more than 5 days (e.g. timedelta difference > 5 days).
4. Dates in each set are not shuffled (strictly monotonically increasing).
"""

import sys
from pathlib import Path
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.split import get_splits
from src.config import TRAIN_END, VAL_END, GAP_DAYS


@pytest.fixture(scope="module")
def split_frames():
    return get_splits()


def test_no_overlap(split_frames):
    """Assert that there is no date overlap between any of the splits."""
    train_df, val_df, test_df = split_frames
    
    train_dates = set(train_df["date"])
    val_dates = set(val_df["date"])
    test_dates = set(test_df["date"])

    assert len(train_dates.intersection(val_dates)) == 0, "Overlap found between Train and Validation sets!"
    assert len(val_dates.intersection(test_dates)) == 0, "Overlap found between Validation and Test sets!"
    assert len(train_dates.intersection(test_dates)) == 0, "Overlap found between Train and Test sets!"


def test_chronological_ordering(split_frames):
    """Assert that every date in an earlier set is strictly before every date in a later set."""
    train_df, val_df, test_df = split_frames

    max_train = pd.to_datetime(train_df["date"].max())
    min_val = pd.to_datetime(val_df["date"].min())
    max_val = pd.to_datetime(val_df["date"].max())
    min_test = pd.to_datetime(test_df["date"].min())

    assert max_train < min_val, f"Train max date ({max_train}) is not strictly before Validation min date ({min_val})"
    assert max_val < min_test, f"Validation max date ({max_val}) is not strictly before Test min date ({min_test})"


def test_gap_between_sets_is_more_than_5_days(split_frames):
    """
    Assert that the gap between sets is more than 5 days (timedelta difference > 5 days).
    With a 5-day dropped buffer, min(later) - max(earlier) equals 6 days > 5 days.
    """
    train_df, val_df, test_df = split_frames

    max_train = pd.to_datetime(train_df["date"].max())
    min_val = pd.to_datetime(val_df["date"].min())
    gap1_days = (min_val - max_train).days

    assert gap1_days > 5, f"Gap between Train and Validation ({gap1_days} days) is not more than 5 days!"

    max_val = pd.to_datetime(val_df["date"].max())
    min_test = pd.to_datetime(test_df["date"].min())
    gap2_days = (min_test - max_val).days

    assert gap2_days > 5, f"Gap between Validation and Test ({gap2_days} days) is not more than 5 days!"


def test_no_shuffling_strictly_increasing(split_frames):
    """Assert that dates in all sets are not shuffled and strictly monotonically increasing."""
    train_df, val_df, test_df = split_frames

    for name, df in [("Train", train_df), ("Validation", val_df), ("Test", test_df)]:
        date_series = pd.to_datetime(df["date"])
        assert date_series.is_monotonic_increasing, f"{name} set dates are not monotonically increasing (shuffled)!"
        assert date_series.is_unique, f"{name} set contains duplicate dates!"
