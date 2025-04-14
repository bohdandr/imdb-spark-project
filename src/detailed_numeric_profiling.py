import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, count, when
import pandas as pd
import numpy as np
from typing import Optional


def safe_sample(df: DataFrame, max_rows: int = 10) -> Optional[DataFrame]:
    total_count = df.count()
    try:
        if total_count > max_rows:
            print(f"Sampling {max_rows} rows from {total_count} to prevent memory issues...")
            df = df.sample(fraction=max_rows / total_count, seed=42)
        return df
    except Exception as e:
        print(f"Failed to convert: {e}")
        return None

def calculate_numeric_statistics(df: DataFrame):
    """
    Computes basic statistics for numeric columns in the DataFrame.

    Args
    ----------
    df : DataFrame
        The PySpark DataFrame to analyze.

    Returns
    ----------
    dict
        A dictionary containing computed statistics (mean, stddev, min, max).
    """
    # Get numeric columns
    numeric_cols = [field.name for field in df.schema.fields if field.dataType.typeName() in ['integer', 'double', 'long']]

    if not numeric_cols:
        print("No numeric columns found.")
        return {}

    statistics = {}

    # Compute basic statistics
    for col_name in numeric_cols:
        summary = df.select(col_name).describe().toPandas()
        stats = {
            "mean": float(summary.loc[summary['summary'] == 'mean', col_name].values[0]),
            "stddev": float(summary.loc[summary['summary'] == 'stddev', col_name].values[0]),
            "min": float(summary.loc[summary['summary'] == 'min', col_name].values[0]),
            "max": float(summary.loc[summary['summary'] == 'max', col_name].values[0]),
        }

        # Get missing count and percentage
        missing_count = df.filter(col(col_name).isNull()).count()
        total_count = df.count()
        missing_percentage = (missing_count / total_count) * 100

        stats.update({
            "missing_count": missing_count,
            "missing_percentage": missing_percentage
        })

        statistics[col_name] = stats

    return statistics

def plot_histogram_and_boxplot(df: DataFrame, max_rows: int = 1000, save_path: str = None):
    """
    Plots histograms and boxplots for numeric columns in the DataFrame.

    Args
    ----------
    df : DataFrame
        The PySpark DataFrame to analyze.
    save_path : str, optional
        Path to save the plot images (default is None, which shows the plot).
    """

    pdf = safe_sample(df, max_rows).toPandas()

    numeric_cols = [col_name for col_name in pdf.columns if pd.api.types.is_numeric_dtype(pdf[col_name])]

    fig, axes = plt.subplots(len(numeric_cols), 2, figsize=(12, len(numeric_cols) * 6))

    for i, col_name in enumerate(numeric_cols):
        axes[i, 0].hist(pdf[col_name].dropna(), bins=20, color='skyblue', edgecolor='black')
        axes[i, 0].set_title(f'Histogram of {col_name}')
        axes[i, 0].set_xlabel(col_name)
        axes[i, 0].set_ylabel('Frequency')

        sns.boxplot(x=pdf[col_name], ax=axes[i, 1], color='lightgreen')
        axes[i, 1].set_title(f'Boxplot of {col_name}')
        axes[i, 1].set_xlabel(col_name)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.show()

def calculate_outliers_and_skewness(df: DataFrame, max_rows: int = 1000):
    """
    Computes the number of outliers and skewness for each numeric column in the DataFrame.

    Args
    ----------
    df : DataFrame
        The PySpark DataFrame to analyze.

    Returns
    ----------
    dict
        A dictionary containing outlier count and skewness for each numeric column.
    """
    
    df = safe_sample(df, max_rows)
    numeric_cols = [field.name for field in df.schema.fields if field.dataType.typeName() in ['integer', 'double', 'long']]

    outlier_info = {}
    for col_name in numeric_cols:
        data = df.select(col_name).toPandas()[col_name]

        # Compute Q1 (25th percentile), Q3 (75th percentile), and IQR (Interquartile Range)
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1

        # Compute the outlier bounds
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Count the number of outliers
        outliers = data[(data < lower_bound) | (data > upper_bound)]
        outlier_count = outliers.shape[0]

        # Compute skewness
        skewness = data.skew()

        outlier_info[col_name] = {
            "outlier_count": outlier_count,
            "skewness": skewness
        }

    return outlier_info

def detailed_numeric_profiling(df: DataFrame, max_rows: int = 1000, save_path: str = None):
    """
    Computes detailed statistics for numeric columns and generates plots.

    Args
    ----------
    df : DataFrame
        The PySpark DataFrame to analyze.
    save_path : str, optional
        Path to save the plot images (default is None, which shows the plot).
    """
    # Calculate numeric statistics
    stats = calculate_numeric_statistics(df)
    print("\nNumeric Statistics:")
    for col, stat in stats.items():
        print(f"\n{col}:")
        for key, value in stat.items():
            print(f"   {key}: {value}")

    # Plot histograms and boxplots
    plot_histogram_and_boxplot(df, max_rows, save_path)

    # Calculate outliers and skewness
    outliers_and_skewness = calculate_outliers_and_skewness(df, max_rows)
    print("\nOutliers and Skewness:")
    for col, info in outliers_and_skewness.items():
        print(f"\n{col}:")
        for key, value in info.items():
            print(f"   {key}: {value}")

