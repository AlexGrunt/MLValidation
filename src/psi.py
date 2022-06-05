import pyspark.sql.functions as F
from pyspark.ml.feature import Bucketizer, QuantileDiscretizer

def psi(actual, expected, column, num_buckets):
  output_col = column + "_bucket"

  qd = QuantileDiscretizer(
      relativeError=0,
      handleInvalid="keep",
      numBuckets=num_buckets,
      inputCol=column,
      outputCol=output_col,
  )

  bucketizer = qd.fit(expected)

  ref_size = expected.count()
  actual_size = actual.count()

  ref_histogram = (
      bucketizer.transform(expected.select(F.col(column)))
      .groupBy(output_col)
      .agg((F.count(F.lit(1)) / ref_size * 100).alias("ref_ratio"))
  )

  actual_histogram = (
      bucketizer.transform(actual.select(F.col(column)))
      .groupBy(output_col)
      .agg((F.count(F.lit(1)) / actual_size * 100).alias("actual_ratio"))
  )

  joined_df = actual_histogram.join(
      ref_histogram,
      actual_histogram[output_col].eqNullSafe(ref_histogram[output_col]),
      how="outer"
  ).fillna(0.001)

  return get_psi_score(joined_df), joined_df


def psi_categorical(actual, expected, column, eps):
  binned_ref_df = expected.groupBy(column).count()
  binned_df = actual.groupBy(column).count()

  ref_size = binned_ref_df.groupBy().sum("count").collect()[0][0]
  actual_size = binned_df.groupBy().sum("count").collect()[0][0]

  ref_histogram = binned_ref_df.withColumn(
      "ref_ratio",
      (F.col("count") / ref_size * 100)
      ).drop("count")

  actual_histogram = binned_df.withColumn(
      "actual_ratio",
      (F.col("count") / actual_size * 100)
      ).drop("count")

  joined_df = actual_histogram.join(
      ref_histogram.withColumnRenamed(column, f'{column}_ref'),
      F.col(column).eqNullSafe(F.col(f'{column}_ref')),
      how="outer").drop(column, f'{column}_ref').fillna(eps)

  return get_psi_score(joined_df), joined_df



def get_psi_score(df):
  res = (
      df.withColumn("psi",
                    (F.col("actual_ratio") - F.col("ref_ratio"))
                    * F.log(F.col("actual_ratio") / F.col("ref_ratio"))
                    ).agg(F.sum("psi").alias("psi")).select("psi").first()[0]
  )

  return res if res is not None else float("inf")
