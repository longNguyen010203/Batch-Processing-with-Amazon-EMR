import argparse
from datetime import datetime

from pyspark.sql import SparkSession
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql.window import Window



def transform_calc_data(data_source: str, output_uri: str) -> None:
    """
    Processes sample food establishment inspection data and queries the data to find 
    the top 10 establishments with the most Red violations from 2006 to 2020.

    :param data_source: The URI of your food establishment data CSV, 
    such as 's3://DOC-EXAMPLE-BUCKET/food-establishment-data.csv'.
    :param output_uri: The URI where output is written, such as 
    's3://DOC-EXAMPLE-BUCKET/restaurant_violation_results'.
    """
    
    with SparkSession.builder.appName("emr-cluster-{}".format(
        datetime.today())).getOrCreate() as spark:
                                 
        # Load the Online Retail CSV data
        if data_source is not None:
            df:DataFrame = spark.read.csv(data_source, header=True, inferSchema=True)
        # Log into EMR stdout
        print(f"Dataset have shape: {(df.count(), df.columns)}")
        
        # Rename Columns
        col_renames = {
            'InvoiceNo': 'OrderID', 
            'StockCode': 'ProductID', 
            'InvoiceDate': 'OrderDate'
        }
        for old_name, new_name in col_renames.items():
            df = df.withColumnRenamed(old_name, new_name)
        
        # Remove spaces
        df = df.withColumn("OrderDate", F.trim(F.col("OrderDate"))) 
            
        # Change data type for column OrderDate
        DATE_FORMAT = ["M/d/yyyy H:mm", "M/d/yyyy H:mm", "M/d/yyyy H:mm"]
        df = df.withColumn(
            "OrderDate",
            F.coalesce(
                F.to_timestamp(F.col("OrderDate"), DATE_FORMAT[0]),
                F.to_timestamp(F.col("OrderDate"), DATE_FORMAT[1]),
                F.to_timestamp(F.col("OrderDate"), DATE_FORMAT[2])
            )
        )
        # Feature Engineering
        TIME_FORMAT = "HH:mm"
        df = df.withColumn("Day", F.dayofmonth("OrderDate"))
        df = df.withColumn("Month", F.month("OrderDate"))
        df = df.withColumn("Year", F.year("OrderDate"))
        df = df.withColumn("HourMinute", F.date_format("OrderDate", TIME_FORMAT))
            
        # Find Start value for CustomerID
        max_customer_id = df.agg(F.max("CustomerID")).collect()[0][0]
        
        # Get a list of OrderIDs with a missing (null) CustomerID value
        order_ids_with_null_customer = df.filter(F.col("CustomerID").isNull()).select("OrderID").distinct()
        
        # Create a sequence number column for OrderIDs with null CustomerID value
        window_spec = Window.orderBy("OrderID")
        order_ids_with_new_customer = order_ids_with_null_customer.withColumn(
            "new_CustomerID", F.row_number().over(window_spec) + max_customer_id)
        
        # Replace null values ​​of the CustomerID column with new values ​​based on OrderID
        df = df.join(order_ids_with_new_customer, "OrderID", "left").withColumn(
            "CustomerID", F.coalesce(F.col("CustomerID"), F.col("new_CustomerID"))
        ).drop("new_CustomerID")
        
        # Use na.fill() to replace null values ​​of the Description column
        df = df.withColumn("Description", F.lower(F.col("Description")))
        df = df.na.fill({"Description": "unknown"})
        
        # Drop NA and Duplicate
        df = df.dropna()
        df = df.dropDuplicates()
        
        # Log into EMR stdout
        print(f"Dataset have shape: {(df.count(), df.columns)}")
        
        # Write our results as parquet files
        df.write.option("header", "true").mode("overwrite").parquet(output_uri)
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_source', help="The URI for you Parquet online retail data, like an S3 bucket location.")
    parser.add_argument('--output_uri', help="The URI where output is saved, like an S3 bucket location.")
    args = parser.parse_args()

    transform_calc_data(args.data_source, args.output_uri)