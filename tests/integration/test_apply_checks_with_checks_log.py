import json
import logging
from decimal import Decimal

import pytest
import pyspark.sql.functions as F
from pyspark.sql import DataFrame, SparkSession

from databricks.labs.dqx import check_funcs
from databricks.labs.dqx.checks_serializer import deserialize_checks
from databricks.labs.dqx.config import ExtraParams, InputConfig, OutputConfig, TableChecksStorageConfig
from databricks.labs.dqx.engine import DQEngine
from databricks.labs.dqx.metrics_observer import DQMetricsObserver
from databricks.labs.dqx.rule import DQRowRule
from databricks.labs.dqx.rule_fingerprint import compute_rule_set_fingerprint_by_metadata
from databricks.labs.dqx.schema import dq_checks_schema

from tests.constants import TEST_CATALOG

INPUT_SCHEMA = "id int, a int, b int, region string"
INPUT_ROWS = [
    [1, 1, None, "EU"],
    [2, None, -1, "US"],
    [3, -5, 2, None],
]

CHECKS = [
    {"name": "a_not_null", "criticality": "error", "check": {"function": "is_not_null", "arguments": {"column": "a"}}},
    {
        "name": "b_not_null_eu",
        "criticality": "warn",
        "filter": "region = 'EU'",
        "check": {"function": "is_not_null", "arguments": {"column": "b"}},
    },
    {
        "criticality": "error",
        "filter": "id > 1",
        "check": {"function": "is_not_less_than", "for_each_column": ["a", "b"], "arguments": {"limit": 0}},
    },
    {
        "name": "missing_column",
        "criticality": "error",
        "check": {"function": "is_not_null", "arguments": {"column": "zzz"}},
    },
    {
        "name": "id_unique",
        "criticality": "error",
        "filter": "region IS NOT NULL",
        "check": {"function": "is_unique", "arguments": {"columns": ["id"]}},
    },
]

# Rules in deserialization order: a_not_null, b_not_null_eu, a_not_less_than, b_not_less_than, missing_column,
# id_unique.
A_NOT_NULL, B_NOT_NULL_EU, A_NOT_LESS, B_NOT_LESS, MISSING, ID_UNIQUE = (
    rule.rule_fingerprint for rule in deserialize_checks(CHECKS)
)

EXPECTED_NOT_APPLIED = {
    1: {A_NOT_LESS, B_NOT_LESS, MISSING},  # id > 1 is false
    2: {B_NOT_NULL_EU, MISSING},  # region is not EU
    3: {B_NOT_NULL_EU, MISSING, ID_UNIQUE},  # region is null: both filters are null
}

EXPECTED_STATUS = {
    1: {
        A_NOT_NULL: "passed",
        B_NOT_NULL_EU: "failed_warning",
        A_NOT_LESS: "not_applied",
        B_NOT_LESS: "not_applied",
        MISSING: "not_applied",
        ID_UNIQUE: "passed",
    },
    2: {
        A_NOT_NULL: "failed_error",
        B_NOT_NULL_EU: "not_applied",
        A_NOT_LESS: "passed",
        B_NOT_LESS: "failed_error",
        MISSING: "not_applied",
        ID_UNIQUE: "passed",
    },
    3: {
        A_NOT_NULL: "passed",
        B_NOT_NULL_EU: "not_applied",
        A_NOT_LESS: "failed_error",
        B_NOT_LESS: "passed",
        MISSING: "not_applied",
        ID_UNIQUE: "not_applied",
    },
}

# Per-row check status, as documented in the query results cookbook.
PER_ROW_STATUS_SQL = """
WITH logged_rules AS (
  SELECT DISTINCT
    log.rule_set_fingerprint,
    rule.rule_fingerprint,
    log.name,
    log.check.function AS function,
    try_element_at(log.check.for_each_column, rule.pos + 1) AS for_each_column
  FROM {log} AS log
  LATERAL VIEW posexplode(log.expanded_rule_fingerprints) rule AS pos, rule_fingerprint
)
SELECT
  results.id,
  logged_rules.rule_fingerprint,
  logged_rules.name,
  logged_rules.function,
  logged_rules.for_each_column,
  CASE
    WHEN array_contains(results._checks.not_applied, logged_rules.rule_fingerprint) THEN 'not_applied'
    WHEN exists(results._errors, e -> e.rule_fingerprint = logged_rules.rule_fingerprint) THEN 'failed_error'
    WHEN exists(results._warnings, w -> w.rule_fingerprint = logged_rules.rule_fingerprint) THEN 'failed_warning'
    ELSE 'passed'
  END AS status
FROM {results} AS results
JOIN logged_rules ON logged_rules.rule_set_fingerprint = results._checks.rule_set_fingerprint
"""


@pytest.fixture
def checks_log_location(make_schema, make_random) -> str:
    """Location for a checks log table created by the engine (dropped with the schema)."""
    schema_name = make_schema(catalog_name=TEST_CATALOG).name
    return f"{TEST_CATALOG}.{schema_name}.checks_log_{make_random(8).lower()}"


@pytest.fixture
def input_df(spark: SparkSession) -> DataFrame:
    return spark.createDataFrame(INPUT_ROWS, INPUT_SCHEMA)


def _not_applied_by_id(result: DataFrame, checks_column: str = "_checks") -> dict[int, set[str]]:
    return {row.id: set(row.not_applied) for row in result.select("id", f"{checks_column}.not_applied").collect()}


def _per_row_status(spark: SparkSession, result: DataFrame, log_location: str) -> dict[int, dict[str, str]]:
    result.createOrReplaceTempView("dqx_checks_log_results")
    rows = spark.sql(PER_ROW_STATUS_SQL.format(log=log_location, results="dqx_checks_log_results")).collect()
    status: dict[int, dict[str, str]] = {}
    for row in rows:
        status.setdefault(row.id, {})[row.rule_fingerprint] = row.status
    return status


def test_checks_column_records_not_applied_checks(ws, spark, input_df, checks_log_location):
    engine = DQEngine(ws, spark, extra_params=ExtraParams(checks_log_location=checks_log_location))

    result = engine.apply_checks_by_metadata(input_df, CHECKS)

    assert result.schema["_checks"].dataType == dq_checks_schema
    assert _not_applied_by_id(result) == EXPECTED_NOT_APPLIED
    checks_values = result.select("_checks.run_id", "_checks.rule_set_fingerprint").distinct().collect()
    assert len(checks_values) == 1
    assert (
        checks_values[0].run_id == engine.apply_checks_by_metadata(input_df, CHECKS).select("_checks.run_id").first()[0]
    )
    assert checks_values[0].rule_set_fingerprint == compute_rule_set_fingerprint_by_metadata(CHECKS)
    # not_applied is sorted, so equal sets give equal values
    for row in result.select("_checks.not_applied").collect():
        assert row.not_applied == sorted(row.not_applied)


def test_per_row_status_decodes_through_checks_log(ws, spark, input_df, checks_log_location):
    engine = DQEngine(ws, spark, extra_params=ExtraParams(checks_log_location=checks_log_location))

    result = engine.apply_checks_by_metadata(input_df, CHECKS)

    assert _per_row_status(spark, result, checks_log_location) == EXPECTED_STATUS


def test_rows_not_listing_a_check_match_its_checked_count(ws, spark, input_df, checks_log_location):
    observer = DQMetricsObserver()
    engine = DQEngine(ws, spark, extra_params=ExtraParams(checks_log_location=checks_log_location), observer=observer)

    result, observation = engine.apply_checks_by_metadata(input_df, CHECKS)
    not_applied = _not_applied_by_id(result)  # triggers the action that populates the observation

    check_metrics = json.loads(observation.get["check_metrics"])
    assert len(check_metrics) == len(EXPECTED_STATUS[1])
    for metric in check_metrics:
        applied_rows = sum(1 for fingerprints in not_applied.values() if metric["rule_fingerprint"] not in fingerprints)
        assert applied_rows == metric["checked_count"], metric


def test_skipped_checks_are_not_applied_when_suppressed(ws, spark, input_df, checks_log_location):
    extra_params = ExtraParams(checks_log_location=checks_log_location, suppress_skipped=True)
    engine = DQEngine(ws, spark, extra_params=extra_params)

    result = engine.apply_checks_by_metadata(input_df, CHECKS)

    assert _not_applied_by_id(result) == EXPECTED_NOT_APPLIED
    assert _per_row_status(spark, result, checks_log_location) == EXPECTED_STATUS


def test_checks_log_is_append_only_and_written_once_per_rule_set(ws, spark, input_df, checks_log_location):
    extra_params = ExtraParams(checks_log_location=checks_log_location)
    engine = DQEngine(ws, spark, extra_params=extra_params)

    engine.apply_checks_by_metadata(input_df, CHECKS)
    engine.apply_checks_by_metadata_and_split(input_df, CHECKS)
    DQEngine(ws, spark, extra_params=extra_params).apply_checks_by_metadata(input_df, CHECKS)

    log_df = spark.read.table(checks_log_location)
    assert log_df.count() == len(CHECKS)  # compact source definitions, written once
    properties = {row.key: row.value for row in spark.sql(f"SHOW TBLPROPERTIES {checks_log_location}").collect()}
    assert properties.get("delta.appendOnly") == "true"


def test_checks_log_holds_source_definitions_as_written(ws, spark, input_df, checks_log_location):
    engine = DQEngine(ws, spark, extra_params=ExtraParams(checks_log_location=checks_log_location))
    checks = [
        *CHECKS,
        {"criticality": "warn", "check": {"function": "sql_expression", "arguments": {"expression": "a + b > 0"}}},
    ]

    engine.apply_checks_by_metadata(input_df, checks)

    rule_set_fingerprint = compute_rule_set_fingerprint_by_metadata(checks)
    rows = spark.read.table(checks_log_location).where(F.col("rule_set_fingerprint") == rule_set_fingerprint)
    for_each_row = rows.where(F.size("check.for_each_column") > 0).first()
    assert for_each_row.check["for_each_column"] == ["a", "b"]
    assert for_each_row.expanded_rule_fingerprints == [A_NOT_LESS, B_NOT_LESS]
    loaded = engine.load_checks(
        TableChecksStorageConfig(location=checks_log_location, rule_set_fingerprint=rule_set_fingerprint)
    )
    assert sorted(json.dumps(check["check"], sort_keys=True) for check in loaded) == sorted(
        json.dumps(check["check"], sort_keys=True) for check in checks
    )
    assert compute_rule_set_fingerprint_by_metadata(loaded) == rule_set_fingerprint


def test_rules_built_in_code_are_logged_via_serialization(ws, spark, input_df, checks_log_location, caplog):
    engine = DQEngine(ws, spark, extra_params=ExtraParams(checks_log_location=checks_log_location))
    rules = [
        DQRowRule(
            check_func=check_funcs.is_in_range,
            column="a",
            check_func_kwargs={"min_limit": Decimal("0.5"), "max_limit": Decimal("10.25")},
        ),
        DQRowRule(name="a_plus_one_not_null", check_func=check_funcs.is_not_null, column=F.col("a") + F.lit(1)),
    ]

    with caplog.at_level(logging.WARNING):
        result = engine.apply_checks(input_df, rules)

    rule_set_fingerprint = result.select("_checks.rule_set_fingerprint").first()[0]
    loaded = engine.load_checks(
        TableChecksStorageConfig(location=checks_log_location, rule_set_fingerprint=rule_set_fingerprint)
    )
    assert len(loaded) == 2
    assert compute_rule_set_fingerprint_by_metadata(loaded) == rule_set_fingerprint
    assert "display text" in caplog.text and "a_plus_one_not_null" in caplog.text


def test_save_in_table_logs_under_run_config_name(ws, spark, make_schema, make_random, checks_log_location):
    schema_name = make_schema(catalog_name=TEST_CATALOG).name
    input_table = f"{TEST_CATALOG}.{schema_name}.input_{make_random(8).lower()}"
    output_table = f"{TEST_CATALOG}.{schema_name}.output_{make_random(8).lower()}"
    checks_table = f"{TEST_CATALOG}.{schema_name}.checks_{make_random(8).lower()}"
    spark.createDataFrame(INPUT_ROWS, INPUT_SCHEMA).write.saveAsTable(input_table)
    engine = DQEngine(ws, spark, extra_params=ExtraParams(checks_log_location=checks_log_location))
    engine.save_checks(CHECKS, TableChecksStorageConfig(location=checks_table, run_config_name="orders"))

    engine.apply_checks_by_metadata_and_save_in_table(
        input_config=InputConfig(location=input_table),
        output_config=OutputConfig(location=output_table),
        checks_location=checks_table,
        run_config_name="orders",
    )

    log_df = spark.read.table(checks_log_location)
    assert {row.run_config_name for row in log_df.collect()} == {"orders"}
    assert log_df.count() == len(CHECKS)
    output = spark.read.table(output_table)
    assert _not_applied_by_id(output) == EXPECTED_NOT_APPLIED

    # applying checks loaded from the log itself does not write them again
    engine.apply_checks_by_metadata_and_save_in_table(
        input_config=InputConfig(location=input_table),
        output_config=OutputConfig(location=output_table, mode="overwrite"),
        checks_location=checks_log_location,
        run_config_name="orders",
    )
    assert spark.read.table(checks_log_location).count() == len(CHECKS)


def test_valid_output_keeps_warnings_and_checks(ws, spark, input_df, checks_log_location):
    engine = DQEngine(ws, spark, extra_params=ExtraParams(checks_log_location=checks_log_location))

    valid_df, invalid_df = engine.apply_checks_by_metadata_and_split(input_df, CHECKS)

    assert "_errors" not in valid_df.columns
    assert {"_warnings", "_checks"} <= set(valid_df.columns)
    assert {"_errors", "_warnings", "_checks"} <= set(invalid_df.columns)
    assert valid_df.count() == 0  # every row fails an error check (missing_column is reported as skipped)


def test_checks_log_disabled_adds_no_checks_column(ws, spark, input_df, checks_log_location):
    engine = DQEngine(ws, spark)

    result = engine.apply_checks_by_metadata(input_df, CHECKS)
    valid_df, _ = engine.apply_checks_by_metadata_and_split(input_df, CHECKS)

    assert "_checks" not in result.columns
    assert "_warnings" not in valid_df.columns
    assert not spark.catalog.tableExists(checks_log_location)


def test_custom_checks_column_name_and_empty_checks(ws, spark, input_df, checks_log_location):
    extra_params = ExtraParams(checks_log_location=checks_log_location, result_column_names={"checks": "dq_checks"})
    engine = DQEngine(ws, spark, extra_params=extra_params)

    result = engine.apply_checks_by_metadata(input_df, CHECKS)
    empty_result = engine.apply_checks_by_metadata(input_df, [])

    assert _not_applied_by_id(result, "dq_checks") == EXPECTED_NOT_APPLIED
    assert empty_result.schema["dq_checks"].dataType == dq_checks_schema
    empty_values = empty_result.select("dq_checks.rule_set_fingerprint", "dq_checks.not_applied").distinct().collect()
    assert [(row.rule_set_fingerprint, row.not_applied) for row in empty_values] == [(None, [])]


def test_log_checks_with_custom_check_function(ws, spark, checks_log_location):
    engine = DQEngine(ws, spark, extra_params=ExtraParams(checks_log_location=checks_log_location))
    checks = [{"criticality": "error", "check": {"function": "a_is_positive", "arguments": {"column": "a"}}}]

    def a_is_positive(column: str):
        return check_funcs.make_condition(F.col(column) <= 0, "not positive", f"{column}_is_not_positive")

    rule_set_fingerprint = engine.log_checks(checks, "custom", {"a_is_positive": a_is_positive})

    row = spark.read.table(checks_log_location).first()
    assert (row.run_config_name, row.rule_set_fingerprint) == ("custom", rule_set_fingerprint)
    assert row.check["function"] == "a_is_positive"
