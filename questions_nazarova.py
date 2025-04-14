from pyspark.sql import SparkSession
from pyspark.sql.functions import col, split, explode, avg, count, row_number, rank, dense_rank
from pyspark.sql.window import Window
import matplotlib.pyplot as plt


def create_spark_session(app_name: str = "IMDb Multi-Join Analysis") -> SparkSession:
    return SparkSession.builder.appName(app_name).getOrCreate()

def load_data(spark, file_path):
    return spark.read.csv(
        file_path,
        sep="\t",
        header=True,
        inferSchema=True,
        nullValue="\\N"
    )

def question_1_high_rated_movies_after_2015(basics_df, ratings_df):
    return basics_df.join(ratings_df, "tconst") \
        .filter((col("startYear") > 2015) & (col("averageRating") > 8.5))

def question_2_recent_action_movies(basics_df):
    return basics_df.filter((col("startYear") >= 2010) & (col("genres").contains("Action")))

def question_3_long_movies(basics_df):
    return basics_df.filter((col("runtimeMinutes") > 150) & (col("titleType") == "movie"))

def question_4_average_rating_by_genre(basics_df, ratings_df):
    df = basics_df.join(ratings_df, "tconst")
    df = df.withColumn("genre", explode(split(col("genres"), ",")))
    return df.groupBy("genre").agg(avg("averageRating").alias("avg_rating"))

def question_5_movies_count_by_year(basics_df):
    return basics_df.groupBy("startYear").count().orderBy(col("startYear"))

def question_6_top_rated_per_year(basics_df, ratings_df):
    df = basics_df.join(ratings_df, "tconst")
    window_spec = Window.partitionBy("startYear").orderBy(col("averageRating").desc())
    return df.withColumn("rank", row_number().over(window_spec)).filter(col("rank") <= 3)

def question_7_top_directors_by_avg_rating(crew_df, ratings_df):
    df = crew_df.join(ratings_df, "tconst")
    df = df.withColumn("director", explode(split(col("directors"), ",")))
    return df.groupBy("director").agg(avg("averageRating").alias("avg_rating")).orderBy(col("avg_rating").desc())

def question_8_most_active_actors_in_comedy(principals_df, basics_df):
    df = principals_df.join(basics_df, "tconst")
    return df.filter(col("category") == "actor") \
             .filter(col("genres").contains("Comedy")) \
             .groupBy("nconst").count().orderBy(col("count").desc())

def question_9_rank_episodes_within_season(basics_df, episode_df, ratings_df):
    df = basics_df.join(episode_df, basics_df.tconst == episode_df.tconst) \
                 .join(ratings_df, basics_df.tconst == ratings_df.tconst)
    window_spec = Window.partitionBy("parentTconst", "seasonNumber").orderBy(col("averageRating").desc())
    return df.withColumn("episode_rank", rank().over(window_spec))

def visualize(df, x_col, y_col, title, kind="bar"):
    pd_df = df.select(x_col, y_col).limit(20).toPandas()
    pd_df.plot(x=x_col, y=y_col, kind=kind, title=title, figsize=(10, 6), legend=False)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def main():
    spark = create_spark_session()

    basics_df = load_data(spark, "data/title.basics.tsv.gz")
    ratings_df = load_data(spark, "data/title.ratings.tsv.gz")
    episode_df = load_data(spark, "data/title.episode.tsv.gz")
    crew_df = load_data(spark, "data/title.crew.tsv.gz")
    principals_df = load_data(spark, "data/title.principals.tsv.gz")
    names_df = load_data(spark, "data/name.basics.tsv.gz")

    print("Question 1: High-rated movies after 2015")
    df1 = question_1_high_rated_movies_after_2015(basics_df, ratings_df)
    df1.show(20)

    print("Question 2: Recent action movies")
    df2 = question_2_recent_action_movies(basics_df)
    df2.show(20)

    print("Question 3: Long movies over 150 minutes")
    df3 = question_3_long_movies(basics_df)
    df3.show(20)

    print("Question 4: Average rating by genre")
    df4 = question_4_average_rating_by_genre(basics_df, ratings_df)
    df4.show(20)
    visualize(df4, "genre", "avg_rating", "Average Rating by Genre")

    print("Question 5: Movie count by release year")
    df5 = question_5_movies_count_by_year(basics_df)
    df5.show(20)
    visualize(df5, "startYear", "count", "Movies Count by Year")

    print("Question 6: Top 3 rated movies per year")
    df6 = question_6_top_rated_per_year(basics_df, ratings_df)
    df6.show(20)

    print("Question 7: Top directors by average rating")
    df7 = question_7_top_directors_by_avg_rating(crew_df, ratings_df)
    df7.show(20)

    print("Question 8: Most active comedy actors")
    df8 = question_8_most_active_actors_in_comedy(principals_df, basics_df)
    df8.show(20)

    print("Question 9: Episode ranking within season")
    df9 = question_9_rank_episodes_within_season(basics_df, episode_df, ratings_df)
    df9.show(20)

if __name__ == "__main__":
    main()