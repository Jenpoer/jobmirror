"""
Monitoring features from test vs trained data
"""

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from scipy.stats import ks_2samp, chisquare, norm
import seaborn as sns

from sklearn.metrics import roc_auc_score, fbeta_score, confusion_matrix, ConfusionMatrixDisplay, accuracy_score

from pyspark.sql.types import NumericType
from pyspark.sql.functions import col
import json

"""
Basic Utilities 
"""

def read_gold_table(table, gold_db, spark):
    """
    Helper function to read all partitions of a gold table
    """
    folder_path = os.path.join(gold_db, table)
    files_list = [os.path.join(folder_path, os.path.basename(f)) for f in glob.glob(os.path.join(folder_path, '*'))]
    df = spark.read.option("header", "true").parquet(*files_list)
    return df

def check_numeric(df, feature_name) -> bool:
    field = df.schema[feature_name]

    return isinstance(field.dataType, NumericType)

def is_one_hot_column(df, column):
    values = df.select(column).distinct().rdd.flatMap(lambda x: x).collect()
    return set(values).issubset({0, 1})

def two_proportion_ztest(x1, n1, x2, n2):
    p1 = x1 / n1
    p2 = x2 / n2
    p_pool = (x1 + x2) / (n1 + n2)

    std_error = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
    z_stat = (p1 - p2) / std_error
    p_value = 2 * norm.sf(abs(z_stat))  # two-tailed

    return z_stat, p_value

"""
Read the feature store
"""

def compare_feature_store(old_features, old_labels, new_features, new_labels):
    # Compare the old spread to the new one
    results = {
        'feature_drift': {},
        'label_drift': {},
        'summary': {
            'features_with_drift': [],
            'label_drift_detected': False,
            'total_features_tested': 0,
            'features_with_significant_drift': 0
        }
    }

    p_value_threshold = 0.05 # Standard p-value check

    old_feature_cols = old_features.columns
    new_feature_cols = new_features.columns
    common_features = list(set(old_feature_cols) & set(new_feature_cols))

    # Go through all the common features
    for feature in common_features:
        old_feature_data = old_features.select(feature)
        new_feature_data = new_features.select(feature)

        # Do a test
        if check_numeric(old_features, feature):
            # DO Kolmogorov–Smirnov (KS) test
            if not is_one_hot_column(old_features, feature):
                # Get the features from the both the old and new dat
                print(f"Numerical Feature : {feature}")
                old_values = [row[0] for row in old_feature_data.collect()]
                new_values = [row[0] for row in new_feature_data.collect()]

                result  = ks_2samp(old_values, new_values)

                # If P-value is below 0.05, drift is statistically significant 
                is_drift = result.pvalue <= p_value_threshold 

                results['summary']['total_features_tested'] += 1
                results['feature_drift'][feature] = is_drift

                if is_drift:
                    # Update values
                    results['summary']['features_with_drift'].append(feature)
                    results['summary']['features_with_significant_drift'] += 1
            else:
                # Do the z-test for OHE columns
                old_vals = old_feature_data.select(feature).rdd.flatMap(lambda x: x).collect()
                new_vals = new_feature_data.select(feature).rdd.flatMap(lambda x: x).collect()

                stat, pval = two_proportion_ztest(old_vals.sum(), len(old_vals), new_vals.sum(), len(new_vals))

                is_drift = pval <= p_value_threshold 

                results['summary']['total_features_tested'] += 1
                results['feature_drift'][feature] = is_drift

                if is_drift:
                    # Update values
                    results['summary']['features_with_drift'].append(feature)
                    results['summary']['features_with_significant_drift'] += 1

        else:
            # Categorical, Chi-Squared Test
            print(f"Categorical Feature : {feature}")
            freq_old = (
                old_feature_data.groupBy(feature)
                .count()
                .withColumn("proportion", col("count") / old_feature_data.count())
                .select(feature, "proportion")
                .toPandas()
                .set_index(feature)["proportion"]
            )

            freq_new = (
                new_feature_data.groupBy(feature)
                .count()
                .withColumn("proportion", col("count") / old_feature_data.count())
                .select(feature, "proportion")
                .toPandas()
                .set_index(feature)["proportion"]
            )

            all_categories = sorted(set(freq_old.index).union(set(freq_new.index)))
            freq_old = freq_old.reindex(all_categories, fill_value=0)
            freq_new = freq_new.reindex(all_categories, fill_value=0)

            result = chisquare(f_obs=freq_new + 1e-6, f_exp=freq_old + 1e-6)
            is_drift = result.pvalue <= p_value_threshold

            results['summary']['total_features_tested'] += 1
            results['feature_drift'][feature] = is_drift
            if is_drift:
                # Update values
                results['summary']['features_with_drift'].append(feature)
                results['summary']['features_with_significant_drift'] += 1

    # Go throught the labels, binary so can use z-test
    old_vals = old_labels.select("label").rdd.flatMap(lambda x: x).collect()
    new_vals = new_labels.select("label").rdd.flatMap(lambda x: x).collect()

    stat, pval = two_proportion_ztest(old_vals.sum(), len(old_vals), new_vals.sum(), len(new_vals))

    is_drift = pval <= p_value_threshold 

    results['label_drift']["label"] = is_drift
    results['summary']['label_drift_detected'] = is_drift

    return results
