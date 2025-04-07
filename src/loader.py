from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType


def load_title_basics_data(spark: SparkSession, file_path: str):
    """
    Loads the 'title.basics.tsv' dataset using a defined schema.

    Args
    ----------
    spark : SparkSession
        The active Spark session.
    file_path : str
        The full path to the 'title.basics.tsv' file.

    Returns
    ----------
    DataFrame
        PySpark DataFrame containing the loaded dataset.
    """
    
    schema = StructType([
        StructField("tconst", StringType(), True),
        StructField("titleType", StringType(), True),
        StructField("primaryTitle", StringType(), True),
        StructField("originalTitle", StringType(), True),
        StructField("isAdult", IntegerType(), True),
        StructField("startYear", IntegerType(), True),
        StructField("endYear", IntegerType(), True),
        StructField("runtimeMinutes", IntegerType(), True),
        StructField("genres", StringType(), True),
    ])

    df = spark.read.csv(
        path=file_path,
        sep="\t",
        header=True,
        schema=schema,
        nullValue="\\N"
    )

    return df
