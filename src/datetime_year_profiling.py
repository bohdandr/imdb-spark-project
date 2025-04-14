import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, min as spark_min, max as spark_max, year
import pandas as pd
import datetime


def profile_year_columns(df: DataFrame, candidate_cols: list = None):
    """
    Profiles columns representing years (e.g., startYear, endYear).

    Args
    ----------
    df : DataFrame
        PySpark DataFrame containing potential year columns.
    candidate_cols : list, optional
        List of column names to treat as year columns. If None, auto-detects common names.
    """
    default_candidates = ["startYear", "endYear", "birthYear", "deathYear", "year"]
    cols_to_check = candidate_cols or [col_name for col_name in df.columns if col_name in default_candidates]

    if not cols_to_check:
        print("No year/datetime-like columns found.")
        return

    for col_name in cols_to_check:
        print(f"\nProfiling year column: {col_name}")

        col_df = df.select(col_name).filter(col(col_name).isNotNull())

        try:
            data = col_df.toPandas()[col_name].astype(float).dropna().astype(int)
        except Exception as e:
            print(f"Could not convert column '{col_name}' to int for plotting: {e}")
            continue

        plt.figure(figsize=(10, 5))
        sns.histplot(data, bins=30, kde=False, color='skyblue')
        plt.title(f"Distribution of {col_name}")
        plt.xlabel("Year")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        data_decade = (data // 10) * 10
        decade_counts = data_decade.value_counts().sort_index()
        plt.figure(figsize=(10, 4))
        sns.barplot(x=decade_counts.index.astype(str), y=decade_counts.values, color="salmon")
        plt.title(f"Counts per decade for {col_name}")
        plt.xticks(rotation=45)
        plt.xlabel("Decade")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.show()

        # Show earliest/latest values
        print(f"  Min year: {data.min()}")
        print(f"  Max year: {data.max()}")

        # Anomaly check: future years
        current_year = datetime.datetime.now().year
        future_years = data[data > current_year]
        if not future_years.empty:
            print(f"  ⚠ Found {len(future_years)} future dates (after {current_year})")

        # Anomaly check for known pairs like birth > death
        if col_name == "birthYear" and "deathYear" in df.columns:
            df_birth_death = df.select("birthYear", "deathYear").filter(
                (col("birthYear").isNotNull()) & (col("deathYear").isNotNull())
            )
            birth_death_pdf = df_birth_death.toPandas().dropna()
            invalid = birth_death_pdf[birth_death_pdf["birthYear"] > birth_death_pdf["deathYear"]]
            if not invalid.empty:
                print(f"  ⚠ Found {len(invalid)} rows where birthYear > deathYear")

