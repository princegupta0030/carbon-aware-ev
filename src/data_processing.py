import pandas as pd
import os
import logging
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_timestamp, date_format

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def preprocess_data_pandas(data_dir="data"):
    """
    Fallback Pandas implementation for data preprocessing if PySpark fails.
    Reads generation and weather data, aligns time formats, merges, and aggregates to hourly.
    """
    gen_path = os.path.join(data_dir, "Plant_1_Generation_Data.csv")
    weather_path = os.path.join(data_dir, "Plant_1_Weather_Sensor_Data.csv")

    logger.info("Loading generation and weather data via Pandas...")
    gen_df = pd.read_csv(gen_path)
    weather_df = pd.read_csv(weather_path)

    # Convert DATE_TIME to consistent datetime format
    # Generation data: 15-05-2020 00:00 (dd-mm-yyyy HH:MM)
    gen_df['DATE_TIME'] = pd.to_datetime(gen_df['DATE_TIME'], format='%d-%m-%Y %H:%M')
    # Weather data: 2020-05-15 00:00:00 (yyyy-mm-dd HH:MM:SS)
    weather_df['DATE_TIME'] = pd.to_datetime(weather_df['DATE_TIME'])

    # Aggregate Generation Data at the Plant level (sum across inverters per timestamp)
    gen_agg = gen_df.groupby('DATE_TIME')[['DC_POWER', 'AC_POWER', 'DAILY_YIELD', 'TOTAL_YIELD']].sum().reset_index()

    # Merge weather and generation data on DATE_TIME
    merged_df = pd.merge(gen_agg, weather_df[['DATE_TIME', 'AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION']], on='DATE_TIME', how='inner')

    # Aggregate to hourly intervals (mean for weather, sum for power)
    merged_df.set_index('DATE_TIME', inplace=True)
    hourly_df = merged_df.resample('H').agg({
        'DC_POWER': 'sum',
        'AC_POWER': 'sum',
        'DAILY_YIELD': 'last', # Keep last value for cumulative
        'TOTAL_YIELD': 'last',
        'AMBIENT_TEMPERATURE': 'mean',
        'MODULE_TEMPERATURE': 'mean',
        'IRRADIATION': 'sum' # Total irradiation per hour
    }).dropna().reset_index()

    logger.info(f"Pandas preprocessing complete. Data shape: {hourly_df.shape}")
    return hourly_df


def preprocess_data_pyspark(data_dir="data"):
    """
    Demonstrates Big Data processing using PySpark.
    Reads, joins, and aggregates data to hourly intervals.
    """
    gen_path = os.path.join(data_dir, "Plant_1_Generation_Data.csv")
    weather_path = os.path.join(data_dir, "Plant_1_Weather_Sensor_Data.csv")

    try:
        # Initialize Spark Session (configure for local machine limits)
        spark = SparkSession.builder \
            .appName("SolarDataProcessing") \
            .master("local[*]") \
            .config("spark.driver.memory", "2g") \
            .getOrCreate()

        logger.info("Spark session initialized. Loading data...")

        # Load data
        gen_df = spark.read.csv(gen_path, header=True, inferSchema=True)
        weather_df = spark.read.csv(weather_path, header=True, inferSchema=True)

        # Fix DATE_TIME format to standard timestamp
        # Generation: "dd-MM-yyyy HH:mm" -> spark uses format "dd-MM-yyyy HH:mm"
        gen_df = gen_df.withColumn("timestamp", to_timestamp(col("DATE_TIME"), "dd-MM-yyyy HH:mm"))
        # Weather: "yyyy-MM-dd HH:mm:ss"
        weather_df = weather_df.withColumn("timestamp", to_timestamp(col("DATE_TIME"), "yyyy-MM-dd HH:mm:ss"))

        # Aggregate generation across inverters
        gen_agg = gen_df.groupBy("timestamp").sum("DC_POWER", "AC_POWER") \
            .withColumnRenamed("sum(DC_POWER)", "DC_POWER") \
            .withColumnRenamed("sum(AC_POWER)", "AC_POWER")

        # Select necessary weather columns
        weather_sel = weather_df.select("timestamp", "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION")

        # Join datasets
        joined_df = gen_agg.join(weather_sel, on="timestamp", how="inner")

        # Extract hour floor for hourly aggregation
        hourly_df = joined_df.withColumn("hourly_timestamp", date_format(col("timestamp"), "yyyy-MM-dd HH:00:00"))

        # Aggregate by hour
        final_hourly_df = hourly_df.groupBy("hourly_timestamp").agg(
            {"DC_POWER": "sum", "AC_POWER": "sum",
             "AMBIENT_TEMPERATURE": "mean", "MODULE_TEMPERATURE": "mean",
             "IRRADIATION": "sum"}
        ).withColumnRenamed("sum(DC_POWER)", "DC_POWER") \
         .withColumnRenamed("sum(AC_POWER)", "AC_POWER") \
         .withColumnRenamed("avg(AMBIENT_TEMPERATURE)", "AMBIENT_TEMPERATURE") \
         .withColumnRenamed("avg(MODULE_TEMPERATURE)", "MODULE_TEMPERATURE") \
         .withColumnRenamed("sum(IRRADIATION)", "IRRADIATION") \
         .withColumnRenamed("hourly_timestamp", "DATE_TIME")

        # Convert back to Pandas for the rest of the workflow
        pandas_df = final_hourly_df.toPandas()

        # Clean up column types
        pandas_df['DATE_TIME'] = pd.to_datetime(pandas_df['DATE_TIME'])
        pandas_df.sort_values('DATE_TIME', inplace=True)
        pandas_df.reset_index(drop=True, inplace=True)

        spark.stop()
        logger.info(f"PySpark preprocessing complete. Data shape: {pandas_df.shape}")
        return pandas_df

    except Exception as e:
        logger.error(f"PySpark processing failed: {e}")
        logger.info("Falling back to Pandas implementation.")
        if 'spark' in locals():
            spark.stop()
        return preprocess_data_pandas(data_dir)

def load_and_preprocess(data_dir="data", output_path="data/processed_hourly.csv"):
    """
    Main entry point for data processing. Uses PySpark with Pandas fallback.
    Saves the output to a CSV.
    """
    df = preprocess_data_pyspark(data_dir)
    df.to_csv(output_path, index=False)
    logger.info(f"Processed data saved to {output_path}")
    return df

if __name__ == "__main__":
    load_and_preprocess()
