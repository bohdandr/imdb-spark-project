from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode, rank, dense_rank, split, size, percent_rank, lead
from pyspark.sql.window import Window
from pyspark.sql import functions as F

def create_spark_session(app_name: str = "IMDb Data Analysis") -> SparkSession:
    """
    Initializes and returns a SparkSession with the specified application name.
    """
    return SparkSession.builder.appName(app_name).getOrCreate()

def load_data(spark: SparkSession, file_path: str):
    """
    Loads TSV data from the given file path using Spark.
    """
    return spark.read.csv(file_path, sep="\t", header=True, inferSchema=True, nullValue="\\N")

# 1. Top 3 Longest Movies per Decade
def top_3_longest_movies_per_decade(df):
    """
    Finds the top 3 longest movies by runtime per decade using window functions.
    """
    df_decade = df.withColumn("decade", F.floor(F.col("startYear") / 10) * 10)
    window_spec = Window.partitionBy("decade").orderBy(F.col("runtimeMinutes").desc())
    return df_decade.filter(F.col("titleType") == "movie") \
                    .withColumn("rank", rank().over(window_spec)) \
                    .filter(F.col("rank") <= 3) \
                    .select("decade", "primaryTitle", "runtimeMinutes", "rank")

# 2. Most Common Genre Pairings
def most_common_genre_pairings(df):
    """
    Identifies the most frequent genre pairings from the dataset.
    """
    df_with_genre = df.withColumn("genre", F.split(F.col("genres"), ","))
    df_exploded = df_with_genre.withColumn("genre", F.explode(F.col("genre")))
    window_spec = Window.partitionBy("tconst").orderBy("genre")
    df_pairs = df_exploded.withColumn("pair", F.concat(F.col("genre"), F.lit(","), F.lead("genre").over(window_spec)))
    df_pairs = df_pairs.filter(F.col("pair").isNotNull())
    return df_pairs.groupBy("pair").count().orderBy(F.col("count").desc())

# 3. Year with the Most Released Titles
def year_with_most_released_titles(df):
    """
    Returns the year with the highest number of released titles.
    """
    return df.groupBy("startYear").count().orderBy(col("count").desc()).limit(1)

# 4. Titles with Runtime Equal to the Median for Their Genre
def titles_equal_to_median_runtime(df):
    """
    Finds titles whose runtime is exactly equal to the median runtime of their genre.
    """
    df_with_genre = df.withColumn("genre", F.split(F.col("genres"), ","))
    df_exploded = df_with_genre.withColumn("genre", F.explode(F.col("genre")))
    median_runtime = df_exploded.groupBy("genre").agg(
        F.expr("percentile_approx(runtimeMinutes, 0.5)").alias("median_runtime")
    )
    df_with_median = df_exploded.join(median_runtime, on="genre", how="inner")
    return df_with_median.filter(F.col("runtimeMinutes") == F.col("median_runtime"))

# 5. Detect Gaps in Movie Releases (Years Where No Movies Were Released)
def detect_gaps_in_movie_releases(spark, df, start_year, end_year):
    """
    Identifies years within a given range where no movies were released.
    """
    all_years = spark.range(start_year, end_year + 1).toDF("startYear")
    movie_years = df.filter(col("titleType") == "movie").select("startYear").distinct()
    missing_years = all_years.join(movie_years, on="startYear", how="left_anti")
    return missing_years

# 6. Titles with the Most Genres (Highest Count of Genres in a Single Row)
def titles_with_most_genres(df):
    """
    Identifies titles that have the largest number of genres listed.
    """
    df_with_genres = df.withColumn("genre_count", size(split(col("genres"), ",")))
    return df_with_genres.orderBy(col("genre_count").desc()).select("primaryTitle", "genre_count")

# 7. Compare Average Runtime of 'Comedy' vs 'Drama' Titles
def compare_avg_runtime_comedy_vs_drama(df):
    """
    Compares the average runtime between Comedy and Drama genres.
    """
    df_with_genres = df.withColumn("genre", split(col("genres"), ","))
    df_exploded = df_with_genres.withColumn("genre", explode(col("genre")))
    comedy_drama_df = df_exploded.filter(col("genre").isin("Comedy", "Drama"))
    return comedy_drama_df.groupBy("genre").agg({"runtimeMinutes": "avg"})

# 8. Titles with Runtime Equal to the Median for Their Genre
def titles_with_runtime_equal_to_median(df):
    """
    Repeats median runtime match using a window function approach.
    """
    df_with_genres = df.withColumn("genre", split(col("genres"), ","))
    df_exploded = df_with_genres.withColumn("genre", explode(col("genre")))
    window_spec = Window.partitionBy("genre")
    median_runtime = df_exploded.withColumn("median_runtime", F.expr("percentile_approx(runtimeMinutes, 0.5)").over(window_spec))
    return median_runtime.filter(col("runtimeMinutes") == col("median_runtime"))

# 9. Longest Short Films (Runtime > 20 minutes and titleType = 'short')
def longest_short_films(df):
    """
    Lists short films whose duration exceeds 20 minutes.
    """
    return df.filter((col("titleType") == "short") & (col("runtimeMinutes") > 20)).select("primaryTitle", "runtimeMinutes", "genres")

def main():
    spark = create_spark_session()
    df = load_data(spark, "title.basics.tsv")

    print("1. Top 3 Longest Movies per Decade:")
    top_3_longest_movies_per_decade(df).show()

    print("2. Most Common Genre Pairings:")
    most_common_genre_pairings(df).show()

    print("3. Year with the Most Released Titles:")
    year_with_most_released_titles(df).show()

    print("4. Titles with Runtime Equal to the Median for Their Genre:")
    titles_equal_to_median_runtime(df).show()

    print("5. Detect Gaps in Movie Releases (Years Where No Movies Were Released):")
    detect_gaps_in_movie_releases(spark, df, 1900, 2025).show()

    print("6. Titles with the Most Genres (Highest Count of Genres in a Single Row):")
    titles_with_most_genres(df).show()

    print("7. Compare Average Runtime of 'Comedy' vs 'Drama' Titles:")
    compare_avg_runtime_comedy_vs_drama(df).show()

    print("8. Titles with Runtime Equal to the Median for Their Genre:")
    titles_with_runtime_equal_to_median(df).show()

    print("9. Longest Short Films (Runtime > 20 minutes and titleType = 'short'):")
    longest_short_films(df).show()

if __name__ == "__main__":
    main()
