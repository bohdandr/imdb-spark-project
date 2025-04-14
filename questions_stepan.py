from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, split, explode, avg, count, row_number, desc
from pyspark.sql.window import Window
import matplotlib.pyplot as plt
import pandas as pd


def create_spark_session(app_name: str = "IMDb Analysis") -> SparkSession:
    """
    Create a Spark session with custom memory and shuffle configurations.

    Args:
        app_name (str): Name for the Spark application.

    Returns:
        SparkSession: Configured Spark session.
    """
    return SparkSession.builder \
        .appName(app_name) \
        .config("spark.executor.memory", "4g") \
        .config("spark.driver.memory", "4g") \
        .config("spark.sql.shuffle.partitions", "200") \
        .getOrCreate()

def load_data(spark: SparkSession, file_path: str) -> DataFrame:
    """
    Load a TSV file as a Spark DataFrame.

    Args:
        spark (SparkSession): Spark session.
        file_path (str): Path to the TSV file.

    Returns:
        DataFrame: Loaded Spark DataFrame.
    """
    return spark.read.csv(
        file_path,
        sep="\t",
        header=True,
        inferSchema=True,
        nullValue="\\N"
    )

def question_1_oldest_horror(basics_df: DataFrame) -> DataFrame:
    """Return oldest horror movies."""
    return basics_df.filter((col("genres").contains("Horror")) & (col("titleType") == "movie")) \
                    .orderBy(col("startYear"))

def question_2_short_documentaries(basics_df: DataFrame) -> DataFrame:
    """Return short documentary films under 40 minutes."""
    return basics_df.filter((col("genres").contains("Documentary")) & (col("runtimeMinutes") < 40))

def question_3_actors_died_after_2000(names_df: DataFrame) -> DataFrame:
    """Return actors who died after the year 2000."""
    return names_df.filter((col("primaryProfession").contains("actor")) & (col("deathYear") > 2000))

def question_4_same_director_writer(crew_df: DataFrame, ratings_df: DataFrame) -> DataFrame:
    """Return movies where directors and writers are the same."""
    return crew_df.join(ratings_df, "tconst") \
                  .filter(col("directors") == col("writers"))

def question_5_titles_high_rating(akas_df: DataFrame, ratings_df: DataFrame) -> DataFrame:
    """Return movies with rating > 9 along with titles and regions."""
    return akas_df.join(ratings_df, akas_df.titleId == ratings_df.tconst) \
                  .filter(col("averageRating") > 9) \
                  .select("title", "region", "averageRating")

def question_6_avg_death_age(names_df: DataFrame) -> DataFrame:
    """Compute average death age per profession."""
    return names_df.filter(col("birthYear").isNotNull() & col("deathYear").isNotNull()) \
                   .withColumn("age", col("deathYear") - col("birthYear")) \
                   .withColumn("profession", explode(split(col("primaryProfession"), ","))) \
                   .groupBy("profession").agg(avg("age").alias("avg_age"))

def question_7_role_count(principals_df: DataFrame) -> DataFrame:
    """Count number of roles per category."""
    return principals_df.groupBy("category").count().orderBy(desc("count"))

def question_8_top_actor_per_genre(principals_df: DataFrame, basics_df: DataFrame) -> DataFrame:
    """Find the most frequent actor per genre."""
    df = principals_df.join(basics_df, "tconst") \
                      .filter(col("category") == "actor") \
                      .withColumn("genre", explode(split(col("genres"), ",")))

    genre_actor_count = df.groupBy("genre", "nconst").count()
    window_spec = Window.partitionBy("genre").orderBy(desc("count"))

    return genre_actor_count.withColumn("rank", row_number().over(window_spec)) \
                            .filter(col("rank") == 1)

def question_9_top_episode_per_series(basics_df: DataFrame, episode_df: DataFrame, ratings_df: DataFrame) -> DataFrame:
    """Return top-rated episode per series based on average rating."""
    df = basics_df.join(episode_df, basics_df.tconst == episode_df.tconst) \
                  .join(ratings_df, basics_df.tconst == ratings_df.tconst)
    window_spec = Window.partitionBy("parentTconst").orderBy(desc("averageRating"))
    return df.withColumn("rank", row_number().over(window_spec)).filter(col("rank") == 1)

def show_and_plot(df: DataFrame, title: str, x_col: str, y_col: str) -> None:
    """
    Show top 20 rows and plot a bar chart.

    Args:
        df (DataFrame): Input Spark DataFrame.
        title (str): Title for plot and printed output.
        x_col (str): Column to be used on X-axis.
        y_col (str): Column to be used on Y-axis.
    """
    pdf = df.toPandas().head(20)
    print(f"\n{title}")
    print(pdf)

    plt.figure(figsize=(10, 6))
    plt.barh(pdf[x_col].astype(str), pdf[y_col])
    plt.xlabel(y_col)
    plt.ylabel(x_col)
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()


def main() -> None:
    spark = create_spark_session()

    basics_df = load_data(spark, "data/title.basics.tsv")
    ratings_df = load_data(spark, "data/title.ratings.tsv")
    episode_df = load_data(spark, "data/title.episode.tsv")
    crew_df = load_data(spark, "data/title.crew.tsv")
    principals_df = load_data(spark, "data/title.principals.tsv")
    akas_df = load_data(spark, "data/title.akas.tsv")
    names_df = load_data(spark, "data/name.basics.tsv")

    show_and_plot(question_6_avg_death_age(names_df), "Average Death Age by Profession", "profession", "avg_age")
    show_and_plot(question_7_role_count(principals_df), "Role Count by Category", "category", "count")
    show_and_plot(question_8_top_actor_per_genre(principals_df, basics_df), "Top Actor per Genre", "genre", "count")


if __name__ == "__main__":
    main()
