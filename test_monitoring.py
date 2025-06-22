"""
Testing file for monitoring file
"""

import os
from utils.metrics_and_monitoring.compare_gt_pred import get_model_metrics
from utils.metrics_and_monitoring.monitoring_features import compare_feature_store, read_gold_table

if __name__ == "__main__":
    print('\n\n---starting job---\n\n')

    # Initialize SparkSession
    spark = pyspark.sql.SparkSession.builder \
        .appName("dev") \
        .master("local[*]") \
        .getOrCreate()
    
    # Set log level to ERROR to hide warnings
    spark.sparkContext.setLogLevel("ERROR")


    home_dir            = os.getcwd()
    prediction_file_dir = os.path.join(home_dir, "datamart", "gold", "prediction_store")
    ground_truth_file_dir = os.path.join(home_dir, "datamart", "gold", "label_store")
    feature_store_dir     = os.path.join(home_dir, "datamart", "gold", "feature_store")
    
    # Read the data from the file dir
    gold_dir = os.path.join(os.getcwd(), "datamart", "gold")
    old_features = read_gold_table("feature_store", gold_dir, spark)
    old_labels   = read_gold_table("label_store", gold_dir, spark)

    # Read the predictions and the ground truth for the future predictions
    # TODO

    # Compare the new features and the new labels the old features and the old labels
    # TODO

    spark.stop()

    print('\n\n---completed job---\n\n')

    