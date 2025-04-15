from pyspark.sql import SparkSession
from pyspark.sql.functions import col, split, explode, avg, count, row_number, rank, dense_rank, desc
from pyspark.sql.window import Window
import matplotlib.pyplot as plt
import pandas as pd

def create_spark_session(app_name: str = "IMDb Extra Questions") -> SparkSession:
    return SparkSession.builder.appName(app_name).getOrCreate()

def load_data(spark, file_path):
    return spark.read.csv(
        file_path,
        sep="\t",
        header=True,
        inferSchema=True,
        nullValue="\\N"
    )

def question_1_long_movies(basics_df):
    return basics_df.filter(col("runtimeMinutes") > 150)

def question_2_actors_multiple_roles(principals_df):
    return principals_df.filter(col("category") == "actor").groupBy("nconst").count().filter(col("count") > 1)

def question_3_died_young(names_df):
    return names_df.filter((col("deathYear").isNotNull()) & (col("birthYear").isNotNull()) & ((col("deathYear") - col("birthYear")) < 40))

def question_4_avg_rating_by_type(basics_df, ratings_df):
    joined = basics_df.join(ratings_df, basics_df.tconst == ratings_df.tconst)
    return joined.groupBy("titleType").agg(avg("averageRating").alias("avgRating")).orderBy(desc("avgRating"))

def question_5_top_rated_alternative_titles(akas_df, ratings_df):
    top_rated = ratings_df.filter(col("averageRating") >= 9)
    return akas_df.join(top_rated, akas_df.titleId == top_rated.tconst).select("titleId", "title", "region", "averageRating")

def question_6_episodes_per_season(episode_df):
    return episode_df.groupBy("parentTconst", "seasonNumber").count().orderBy("parentTconst", "seasonNumber")

def question_7_avg_runtime_by_genre(basics_df):
    df = basics_df.withColumn("genre", explode(split(col("genres"), ",")))
    return df.groupBy("genre").agg(avg("runtimeMinutes").alias("avgRuntime")).orderBy(desc("avgRuntime"))

def question_8_best_movie_per_genre(basics_df, ratings_df):
    df = basics_df.join(ratings_df, "tconst").withColumn("genre", explode(split(col("genres"), ",")))
    window_spec = Window.partitionBy("genre").orderBy(desc("averageRating"))
    return df.withColumn("rank", row_number().over(window_spec)).filter(col("rank") == 1)

def question_9_top_tvseries_by_votes(basics_df, ratings_df):
    df = basics_df.filter(col("titleType") == "tvSeries").join(ratings_df, "tconst")
    window_spec = Window.orderBy(desc("numVotes"))
    return df.withColumn("rank", row_number().over(window_spec)).filter(col("rank") <= 10)

def question_10_movies_per_genre(basics_df):
    df = basics_df.withColumn("genre", explode(split(col("genres"), ",")))
    return df.groupBy("genre").count().orderBy(desc("count"))

def show_and_plot(df, title, x_col, y_col):
    pdf = df.toPandas().head(20)
    print(title)
    print(pdf)
    plt.figure(figsize=(10, 6))
    plt.barh(pdf[x_col].astype(str), pdf[y_col])
    plt.xlabel(y_col)
    plt.ylabel(x_col)
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

def main():
    spark = create_spark_session()

    basics_df = load_data(spark, "data/title.basics.tsv.gz")
    ratings_df = load_data(spark, "data/title.ratings.tsv.gz")
    episode_df = load_data(spark, "data/title.episode.tsv.gz")
    crew_df = load_data(spark, "data/title.crew.tsv.gz")
    principals_df = load_data(spark, "data/title.principals.tsv.gz")
    akas_df = load_data(spark, "data/title.akas.tsv.gz")
    names_df = load_data(spark, "data/name.basics.tsv.gz")

    show_and_plot(question_10_movies_per_genre(basics_df), "Movies Per Genre", "genre", "count")
    show_and_plot(question_7_avg_runtime_by_genre(basics_df), "Average Runtime by Genre", "genre", "avgRuntime")
    show_and_plot(question_9_top_tvseries_by_votes(basics_df, ratings_df), "Top 10 TV Series by Votes", "primaryTitle", "numVotes")

    print(question_1_long_movies(basics_df).show(20))
    print(question_2_actors_multiple_roles(principals_df).show(20))
    print(question_3_died_young(names_df).show(20))
    print(question_4_avg_rating_by_type(basics_df, ratings_df).show(20))
    print(question_5_top_rated_alternative_titles(akas_df, ratings_df).show(20))
    print(question_6_episodes_per_season(episode_df).show(20))
    print(question_8_best_movie_per_genre(basics_df, ratings_df).show(20))

if __name__ == "__main__":
    main()
