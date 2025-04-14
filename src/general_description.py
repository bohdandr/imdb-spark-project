import os
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import logging.config
import yaml

from pyspark.sql import DataFrame
from pyspark.sql.functions import col, countDistinct, count, when, desc

from utils.logging_utils import setup_logger

logger = setup_logger("general_description")

def missing_value_summary(df: DataFrame):
    total_rows = df.count()
    logger.info(f"Computing missing values across {total_rows} rows...")

    missing_values = df.select([
        count(when(col(c).isNull(), c)).alias(c) for c in df.columns
    ])
    missing_percentage = df.select([
        ((count(when(col(c).isNull(), c)) / total_rows) * 100).alias(f"{c}_pct_missing")
        for c in df.columns
    ])

    logger.info("Missing Value Counts:")
    logger.info(missing_values)
    missing_values.show()

    logger.info("Missing Value Percentages:")
    logger.info(missing_percentage)
    missing_percentage.show()

def unique_value_counts(df: DataFrame):
    logger.info("Computing unique value counts for all columns...")
    unique_counts = df.select([
        countDistinct(col(c)).alias(c) for c in df.columns
    ])
    logger.info(unique_counts)
    unique_counts.show()

def most_frequent_values(df: DataFrame, top_n=3):
    logger.info(f"Computing top {top_n} most frequent values per column...")

    for column in df.columns:
        logger.info(f"Top {top_n} values for column '{column}':")
        top_values = df.groupBy(column).count().orderBy(desc("count")).limit(top_n)
        logger.info(top_values)
        top_values.show()

def null_histogram(df: DataFrame):
    logger.info("Generating null value histogram...")

    null_counts = [df.filter(col(c).isNull()).count() for c in df.columns]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=df.columns, y=null_counts, palette="viridis")
    plt.title("Number of Nulls per Column")
    plt.xlabel("Columns")
    plt.ylabel("Null Count")
    plt.xticks(rotation=45)
    plt.tight_layout()

    file_path = os.path.join("plots/", "null_values_histogram.png")
    plt.savefig(file_path)
    logger.info(f"Null histogram saved to {file_path}")
    plt.close()

def describe_dataset(df: DataFrame):
    """
    Prints general information about the DataFrame, including missing values,
    unique value counts, and most frequent values.

    Args
    ----------
    df : DataFrame
        The PySpark DataFrame to describe.
    """
    logger.info("Describing dataset schema and structure...")
    df.printSchema()
    row_count = df.count()
    logger.info(f"Number of rows: {row_count}")
    logger.info(f"Number of columns: {len(df.columns)}")
    logger.info(f"Column names: {df.columns}")

    missing_value_summary(df)
    unique_value_counts(df)
    most_frequent_values(df)
    null_histogram(df)