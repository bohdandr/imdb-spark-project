from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode, split, countDistinct, avg, row_number, rank, dense_rank
from pyspark.sql.window import Window


def create_spark_session(app_name: str = "IMDb Multi-Join Analysis") -> SparkSession:
    spark = SparkSession.builder \
        .appName(app_name) \
        .config("spark.executor.memory", "4g") \
        .config("spark.driver.memory", "4g") \
        .config("spark.sql.shuffle.partitions", "200") \
        .getOrCreate()
    return spark

def load_data(spark, path):
    return spark.read.csv(path, sep="\t", header=True, inferSchema=True, nullValue="\\N")


def question_1_ukrainian_titles(akas_df):
    return akas_df.filter(col("language") == "uk")

def question_2_documentaries_short(basics_df):
    return basics_df.filter((col("genres").contains("Documentary")) & (col("runtimeMinutes") < 30))

def question_3_highly_rated_old_movies(basics_df, ratings_df):
    return basics_df.join(ratings_df, "tconst") \
        .filter((col("startYear") < 1980) & (col("averageRating") >= 8.0))


def question_4_directors_with_10_plus_movies(crew_df, basics_df):
    df = crew_df.withColumn("director", explode(split("directors", ","))) \
                .join(basics_df, "tconst") \
                .filter(col("titleType") == "movie")
    return df.groupBy("director").count().filter(col("count") >= 10)

def question_5_female_actors_in_top_rated_movies(principals_df, names_df, ratings_df):
    df = principals_df.join(ratings_df, "tconst") \
                      .join(names_df, "nconst") \
                      .filter((col("category") == "actress") & (col("averageRating") > 8.5))
    return df.select("primaryName", "averageRating").distinct()


def question_6_avg_runtime_per_genre(basics_df):
    df = basics_df.withColumn("genre", explode(split("genres", ","))) \
                  .filter(col("runtimeMinutes").isNotNull())
    return df.groupBy("genre").agg(avg("runtimeMinutes").alias("avg_runtime"))

def question_7_actors_count_per_genre(principals_df, basics_df):
    df = principals_df.join(basics_df, "tconst") \
                      .filter(col("category") == "actor") \
                      .withColumn("genre", explode(split("genres", ",")))
    return df.groupBy("genre").agg(countDistinct("nconst").alias("actor_count"))


def question_8_rank_directors_by_avg_rating(crew_df, ratings_df):
    df = crew_df.withColumn("director", explode(split("directors", ","))) \
                .join(ratings_df, "tconst")
    avg_df = df.groupBy("director").agg(avg("averageRating").alias("avg_rating"))
    window = Window.orderBy(col("avg_rating").desc())
    return avg_df.withColumn("rank", dense_rank().over(window))

def question_9_rank_actors_by_number_of_movies(principals_df):
    df = principals_df.filter(col("category").isin("actor", "actress"))
    actor_counts = df.groupBy("nconst").count()
    window = Window.orderBy(col("count").desc())
    return actor_counts.withColumn("rank", row_number().over(window))


def main():
    spark = create_spark_session()

    basics_df = load_data(spark, "data/title.basics.tsv")
    ratings_df = load_data(spark, "data/title.ratings.tsv")
    akas_df = load_data(spark, "data/title.akas.tsv")
    crew_df = load_data(spark, "data/title.crew.tsv")
    principals_df = load_data(spark, "data/title.principals.tsv")
    names_df = load_data(spark, "data/name.basics.tsv")

    print("Q1: Titles available in Ukrainian")
    question_1_ukrainian_titles(akas_df).show(10)

    print("Q2: Short documentaries")
    question_2_documentaries_short(basics_df).show(10)

    print("Q3: High-rated movies before 1980")
    question_3_highly_rated_old_movies(basics_df, ratings_df).show(10)

    print("Q4: Directors with >=10 movies")
    question_4_directors_with_10_plus_movies(crew_df, basics_df).show(10)

    print("Q5: Female actors in high-rated movies")
    question_5_female_actors_in_top_rated_movies(principals_df, names_df, ratings_df).show(10)

    print("Q6: Avg runtime per genre")
    question_6_avg_runtime_per_genre(basics_df).show(10)

    print("Q7: Actor count per genre")
    question_7_actors_count_per_genre(principals_df, basics_df).show(10)

    print("Q8: Top directors by avg rating (ranked)")
    question_8_rank_directors_by_avg_rating(crew_df, ratings_df).show(10)

    print("Q9: Most active actors (by movie count)")
    question_9_rank_actors_by_number_of_movies(principals_df).show(10)


if __name__ == "__main__":
    main()
