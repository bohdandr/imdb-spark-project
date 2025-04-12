from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode, rank, dense_rank, split
from pyspark.sql.window import Window

def create_spark_session(app_name: str = "IMDb Data Analysis") -> SparkSession:
    return SparkSession.builder.appName(app_name).getOrCreate()

def load_data(spark: SparkSession, file_path: str):
    return spark.read.csv(file_path, sep="\t", header=True, inferSchema=True, nullValue="\\N")

# 1.Movies released after 2000
def movies_after_2000(df):
    return df.filter((df["titleType"] == "movie") & (df["startYear"] > 2000))

# 2.Adult titles longer than 60 minutes
def adult_titles_over_60(df):
    return df.filter((df["isAdult"] == 1) & (df["runtimeMinutes"] > 60))

# 3.Documentaries released before 1950
def documentaries_before_1950(df):
    return df.filter((df["genres"].contains("Documentary")) & (df["startYear"] < 1950))

# 4.Comparison of movies by year
def compare_movies_by_year(df):
    df_a = df.alias("a")
    df_b = df.alias("b")

    result = df_a.join(df_b, (df_a["primaryTitle"] == df_b["primaryTitle"]) & (df_a["startYear"] != df_b["startYear"])) \
                 .select(df_a["tconst"], df_a["primaryTitle"], df_a["startYear"], df_b["startYear"])

    return result

# 5.Movies released in the same year and genre as "Carmencita"
def movies_same_year_genre(df):
    carmencita = df.filter(df["primaryTitle"] == "Carmencita").select("startYear", "genres").first()
    year = carmencita["startYear"]
    genres = carmencita["genres"]
    return df.filter((df["startYear"] == year) & df["genres"].contains(genres))

# 6.Count movies by title type
def count_movies_by_title_type(df):
    return df.groupBy("titleType").count()

# 7.Count movies by year and genre
def count_movies_by_year_and_genre(df):
    df_with_genres = df.withColumn("genre", split(df["genres"], ","))

    return df_with_genres.withColumn("genre", explode(df_with_genres["genre"])) \
                         .groupBy("startYear", "genre") \
                         .count()

# 8.Rank movies based on runtime per year
def rank_movies_per_year(df):
    window_spec = Window.partitionBy("startYear").orderBy(df["runtimeMinutes"].desc())
    return df.withColumn("rank", rank().over(window_spec)).select("tconst", "primaryTitle", "startYear", "runtimeMinutes", "rank")

# 9. Rank movies within each genre based on runtime
def rank_movies_per_genre(df):
    window_spec = Window.partitionBy("genres").orderBy(df["runtimeMinutes"].desc())
    return df.withColumn("rank", dense_rank().over(window_spec)).select("tconst", "primaryTitle", "genres", "runtimeMinutes", "rank")

def main():
    spark = create_spark_session()
    df = load_data(spark, "title.basics.tsv")

    #Movies released after 2000
    print("Movies Released After 2000:")
    movies_after_2000(df).show()

    #Adult titles longer than 60 minutes
    print("Adult Titles Longer Than 60 Minutes:")
    adult_titles_over_60(df).show()

    #Documentaries released before 1950
    print("Documentaries Released Before 1950:")
    documentaries_before_1950(df).show()

    #Comparison of movies by year
    print("Movies with Different Original and Primary Titles:")
    compare_movies_by_year(df).show()

    #Movies released in the same year and genre as "Carmencita"
    print("Movies Released in the Same Year and Genre as 'Carmencita':")
    movies_same_year_genre(df).show()

    #Count movies by title type
    print("Count of Movies by Title Type:")
    count_movies_by_title_type(df).show()

    #Count movies by year and genre
    print("Average Runtime by Genre:")
    count_movies_by_year_and_genre(df).show()

    #Rank movies based on runtime per year
    print("Ranked Movies Based on Runtime per Year:")
    rank_movies_per_year(df).show()

    #Rank movies within each genre based on runtime
    print("Ranked Movies Within Each Genre Based on Runtime:")
    rank_movies_per_genre(df).show()

if __name__ == "__main__":
    main()
