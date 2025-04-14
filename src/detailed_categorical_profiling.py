import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, count, when
import pandas as pd


def calculate_categorical_statistics(df: DataFrame, top_n: int = 10):
    """
    Calculates statistics for categorical/string columns.

    Args
    ----------
    df : DataFrame
        The PySpark DataFrame to analyze.
    top_n : int
        Number of top frequent values to show.

    Returns
    ----------
    dict
        A dictionary with stats for each categorical column.
    """
    categorical_cols = [field.name for field in df.schema.fields if field.dataType.typeName() == 'string']
    stats = {}

    for col_name in categorical_cols:
        print(f"\nColumn: {col_name}")

        total_count = df.count()
        missing_count = df.filter((col(col_name).isNull()) | (col(col_name) == "")).count()
        missing_percentage = (missing_count / total_count) * 100

        unique_count = df.select(col_name).distinct().count()

        top_values = df.groupBy(col_name).count().orderBy('count', ascending=False).limit(top_n).toPandas()

        stats[col_name] = {
            "missing_count": missing_count,
            "missing_percentage": missing_percentage,
            "unique_count": unique_count,
            "top_values": top_values
        }

        print(f" - Missing values: {missing_count} ({missing_percentage:.2f}%)")
        print(f" - Unique categories: {unique_count}")
        print(f" - Top {top_n} most frequent values:")
        print(top_values)

    return stats

def plot_categorical_distributions(stats: dict, save_path_prefix: str = None):
    """
    Plots bar charts for the top frequent values in each categorical column.

    Args
    ----------
    stats : dict
        Output from calculate_categorical_statistics.
    save_path_prefix : str, optional
        Prefix for saving plot images. If None, plots are shown instead.
    """
    for col_name, stat in stats.items():
        top_values = stat["top_values"]
        if top_values.empty:
            continue

        plt.figure(figsize=(10, 6))
        sns.barplot(data=top_values, x="count", y=col_name, palette="pastel")
        plt.title(f"Top {len(top_values)} values for '{col_name}'")
        plt.xlabel("Count")
        plt.ylabel("Category")
        plt.tight_layout()

        if save_path_prefix:
            plt.savefig(f"plots/{save_path_prefix}_{col_name}.png")
        else:
            plt.show()

def detailed_categorical_profiling(df: DataFrame, top_n: int = 10, save_path_prefix: str = None):
    """
    Full profiling pipeline for categorical columns:
    stats + plots.

    Args
    ----------
    df : DataFrame
        The PySpark DataFrame to analyze.
    top_n : int
        Number of top frequent values to include.
    save_path_prefix : str, optional
        Prefix for saving plots. If None, plots will be shown.
    """
    stats = calculate_categorical_statistics(df, top_n=top_n)
    plot_categorical_distributions(stats, save_path_prefix=save_path_prefix)
