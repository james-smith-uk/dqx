from pyspark.sql.types import StructType, StructField, ArrayType, StringType

# Per-row record of the rule set applied to the row, added when a checks log is enabled.
# *not_applied* holds the fingerprints of checks whose filter did not match the row, plus checks skipped
# for the whole run. Every other check of the rule set (as stored in the checks log) was applied to the row.
dq_checks_schema = StructType(
    [
        StructField("run_id", StringType(), nullable=True),
        StructField("rule_set_fingerprint", StringType(), nullable=True),
        StructField("not_applied", ArrayType(StringType()), nullable=True),
    ]
)
