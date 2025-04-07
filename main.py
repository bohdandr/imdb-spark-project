from pyspark.sql import SparkSession
from src.loader import load_title_basics_data


def main():
    spark = SparkSession.builder.appName("LoadTitleBasics").getOrCreate()

    file_path = "title.basics.tsv"
    df = load_title_basics_data(spark, file_path)

    print("Schema:")
    df.printSchema()

    print("Sample rows:")
    df.show(5)

    spark.stop()


if __name__ == "__main__":
    main()
