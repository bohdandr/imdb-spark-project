from pyspark.sql import SparkSession
from pyspark.sql.functions import col, split, explode, avg, count, row_number, rank, dense_rank, desc
from pyspark.sql.window import Window
import matplotlib.pyplot as plt
import pandas as pd

def create_spark_session(app_name: str = "IMDb Extra Questions") -> SparkSession:
    """
    Create and return a new SparkSession with the given application name.
    """
    return SparkSession.builder.appName(app_name).getOrCreate()

def load_data(spark, file_path):
    """
    Load a TSV file into a Spark DataFrame using the given file path.
    Handles null values represented as '\\N'.
    """
    return spark.read.csv(
        file_path,
        sep="\t",
        header=True,
        inferSchema=True,
        nullValue="\\N"
    )

def question_1_recent_comedies(basics_df):
    """
    Filter movies of genre 'Comedy' that started after 2010.
    
    Source: title.basics.tsv
    Purpose: Identify recent comedies.
    """
    return basics_df.filter((col("genres").contains("Comedy")) & (col("startYear") > 2010))

def question_2_people_born_after_1990_alive(names_df):
    """
    Filter people born after 1990 who are still alive.
    
    Source: name.basics.tsv
    Purpose: Identify young potential actors/directors.
    """
    return names_df.filter((col("birthYear") > 1990) & (col("deathYear").isNull()))

def question_3_movies_without_genre(basics_df):
    """
    Filter movies that have no genre specified (null).
    
    Source: title.basics.tsv
    Purpose: Detect incomplete or erroneous records.
    """
    return basics_df.filter(col("genres").isNull())

def question_4_writers_high_rating(crew_df, ratings_df):
    """
    Find writers who worked on movies with an average rating above 8.
    
    Source: title.crew.tsv, title.ratings.tsv
    Purpose: Analyze writers of successful films.
    """
    return crew_df.join(ratings_df, "tconst").filter(col("averageRating") > 8).select("tconst", "writers", "averageRating")

def question_5_us_titles_with_runtime(akas_df, basics_df):
    """
    Get titles and runtimes of movies that have alternative titles in the US.
    
    Source: title.akas.tsv, title.basics.tsv
    Purpose: Explore US market alternative titles.
    """
    return akas_df.filter(col("region") == "US") \
                 .join(basics_df, akas_df.titleId == basics_df.tconst) \
                 .select("title", "runtimeMinutes")

def question_6_movies_per_genre(basics_df):
    """
    Count the number of movies for each genre.
    
    Source: title.basics.tsv
    Purpose: Visualize genre distribution.
    """
    df = basics_df.withColumn("genre", explode(split(col("genres"), ",")))
    return df.groupBy("genre").count().orderBy(desc("count"))

def question_7_people_per_profession(names_df):
    """
    Count the number of people by their profession.
    
    Source: name.basics.tsv
    Purpose: Discover most common roles in the film industry.
    """
    df = names_df.withColumn("profession", explode(split(col("primaryProfession"), ",")))
    return df.groupBy("profession").count().orderBy(desc("count"))

def question_8_top_main_actor(principals_df):
    """
    Find the actor with the highest number of leading roles.
    
    Source: title.principals.tsv
    Purpose: Identify the most active lead actor.
    """
    df = principals_df.filter(col("category") == "actor")
    window_spec = Window.orderBy(desc("count"))
    return df.groupBy("nconst").count().withColumn("rank", row_number().over(window_spec)).filter(col("rank") == 1)

def question_9_top_directors(crew_df):
    """
    Find the top 3 directors by number of directed movies.
    
    Source: title.crew.tsv
    Purpose: Identify the most prolific directors.
    """
    df = crew_df.withColumn("director", explode(split(col("directors"), ",")))
    director_counts = df.groupBy("director").count()
    window_spec = Window.orderBy(desc("count"))
    return director_counts.withColumn("rank", row_number().over(window_spec)).filter(col("rank") <= 3)

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

    show_and_plot(question_1_recent_comedies(basics_df), "Recent Comedies", "primaryTitle", "startYear")
    show_and_plot(question_6_movies_per_genre(basics_df), "Movies Per Genre", "genre", "count")
    show_and_plot(question_9_top_directors(crew_df), "Top 3 Directors by Number of Movies", "director", "count")

    print(question_2_people_born_after_1990_alive(names_df).show(20))
    print(question_3_movies_without_genre(basics_df).show(20))
    print(question_4_writers_high_rating(crew_df, ratings_df).show(20))
    print(question_5_us_titles_with_runtime(akas_df, basics_df).show(20))
    print(question_7_people_per_profession(names_df).show(20))
    print(question_8_top_main_actor(principals_df).show(20))

if __name__ == "__main__":
    main()