from decimal import Decimal
from pathlib import Path
from unittest.mock import MagicMock, create_autospec

import pytest
import yaml
import pyspark.sql.functions as F
from pyspark.sql import Column, DataFrame, SparkSession
from pyspark.sql.conf import RuntimeConfig
from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import NotFound

from databricks.labs.dqx import check_funcs
from databricks.labs.dqx.check_funcs import make_condition
from databricks.labs.dqx.checks_serializer import ChecksNormalizer, deserialize_checks, serialize_checks
from databricks.labs.dqx.config import ExtraParams, TableChecksStorageConfig
from databricks.labs.dqx.engine import DQEngine, DQEngineCore
from databricks.labs.dqx.errors import (
    InvalidCheckError,
    InvalidConfigError,
    InvalidParameterError,
    UnsafeSqlQueryError,
)
from databricks.labs.dqx.rule import DQForEachColRule, DQRowRule, DQRule
from databricks.labs.dqx.rule_fingerprint import compute_rule_set_fingerprint, compute_rule_set_fingerprint_by_metadata

LOG_LOCATION = "catalog.schema.checks_log"
RESOURCES = Path(__file__).parent.parent / "resources"

METADATA_CHECKS = [
    {
        "criticality": "warn",
        "filter": "b > 0",
        "check": {"function": "is_not_null", "for_each_column": ["a", "b"]},
    },
    {"criticality": "error", "check": {"function": "sql_expression", "arguments": {"expression": "a > 1"}}},
]


def _mock_spark(pipeline_id: str | None = None) -> MagicMock:
    spark = create_autospec(SparkSession)
    spark.conf = create_autospec(RuntimeConfig)
    spark.conf.get.return_value = pipeline_id
    return spark


def _logged_rows(spark: MagicMock) -> list[list]:
    """Rows passed to createDataFrame by the checks table writer."""
    assert spark.createDataFrame.call_count == 1
    return spark.createDataFrame.call_args.args[0]


def _mock_input_df(columns: list[str]) -> MagicMock:
    df = create_autospec(DataFrame)
    df.columns = columns
    return df


def test_extra_params_checks_log_location_defaults_to_none():
    assert ExtraParams().checks_log_location is None
    assert ExtraParams(checks_log_location=LOG_LOCATION).checks_log_location == LOG_LOCATION
    assert ExtraParams(checks_log_location="hive_schema.checks_log").checks_log_location == "hive_schema.checks_log"


@pytest.mark.parametrize("location", ["", "checks.yml", "/Volumes/a/b/checks.json", "a.b; drop table x"])
def test_extra_params_rejects_invalid_checks_log_location(location: str):
    with pytest.raises(InvalidParameterError, match="checks_log_location"):
        ExtraParams(checks_log_location=location)


def test_extra_params_rejects_unsafe_checks_log_location():
    with pytest.raises(UnsafeSqlQueryError):
        ExtraParams(checks_log_location="catalog.drop.checks_log")


def test_table_storage_config_rejects_append_only_with_overwrite():
    with pytest.raises(InvalidConfigError, match="append_only"):
        TableChecksStorageConfig(location=LOG_LOCATION, mode="overwrite", append_only=True)
    assert TableChecksStorageConfig(location=LOG_LOCATION, append_only=True).append_only


def test_log_checks_requires_checks_log_location(mock_workspace_client: WorkspaceClient):
    engine = DQEngineCore(mock_workspace_client, _mock_spark())

    assert engine.checks_log_location is None
    with pytest.raises(InvalidConfigError, match="checks_log_location"):
        engine.log_checks(METADATA_CHECKS)


def test_log_checks_rejects_empty_checks(mock_workspace_client: WorkspaceClient):
    engine = DQEngineCore(mock_workspace_client, _mock_spark(), ExtraParams(checks_log_location=LOG_LOCATION))

    with pytest.raises(InvalidParameterError):
        engine.log_checks([])


def test_log_checks_logs_metadata_as_written_to_append_only_table(mock_workspace_client: MagicMock):
    spark = _mock_spark()
    ws = mock_workspace_client
    ws.tables.get.side_effect = NotFound("checks log does not exist yet")
    engine = DQEngine(ws, spark, extra_params=ExtraParams(checks_log_location=LOG_LOCATION))

    rule_set_fingerprint = engine.log_checks(METADATA_CHECKS, run_config_name="orders")

    assert rule_set_fingerprint == compute_rule_set_fingerprint(deserialize_checks(METADATA_CHECKS))
    rows = _logged_rows(spark)
    # compact rows, as written: one row per source check, the for_each_column check not expanded
    assert len(rows) == len(METADATA_CHECKS)
    for_each_row = rows[0]
    assert for_each_row[2]["for_each_column"] == ["a", "b"]
    assert for_each_row[3] == "b > 0"
    assert for_each_row[4] == "orders"
    assert for_each_row[-1] == [rule.rule_fingerprint for rule in deserialize_checks([METADATA_CHECKS[0]])]
    assert all(row[-2] == rule_set_fingerprint for row in rows)
    spark.createDataFrame.return_value.write.saveAsTable.assert_called_once_with(LOG_LOCATION, mode="append")
    statements = [call.args[0] for call in spark.sql.call_args_list]
    assert any(
        "'delta.appendOnly' = 'true'" in stmt and "`catalog`.`schema`.`checks_log`" in stmt for stmt in statements
    )


def test_log_checks_writes_each_rule_set_once_per_engine(mock_workspace_client: WorkspaceClient):
    spark = _mock_spark()
    engine = DQEngineCore(mock_workspace_client, spark, ExtraParams(checks_log_location=LOG_LOCATION))

    first = engine.log_checks(METADATA_CHECKS)
    second = engine.log_checks(METADATA_CHECKS)

    assert first == second
    assert spark.createDataFrame.call_count == 1


def test_log_checks_serializes_rules_built_in_code(mock_workspace_client: WorkspaceClient):
    spark = _mock_spark()
    engine = DQEngineCore(mock_workspace_client, spark, ExtraParams(checks_log_location=LOG_LOCATION))
    rules = [
        DQRowRule(
            check_func=check_funcs.is_in_range,
            column="price",
            check_func_kwargs={"min_limit": Decimal("0.5"), "max_limit": Decimal("10.25")},
        ),
        *DQForEachColRule(check_func=check_funcs.is_not_null, columns=["a", "b"], criticality="warn").get_rules(),
    ]

    rule_set_fingerprint = engine.log_checks(rules)

    assert rule_set_fingerprint == compute_rule_set_fingerprint(rules)
    rows = _logged_rows(spark)
    assert len(rows) == 3  # rules built in code are already expanded
    assert [row[-3] for row in rows] == [rule.rule_fingerprint for rule in rules]


def _custom_is_positive(column: str) -> Column:
    return make_condition(F.col(column) <= 0, "not positive", f"{column}_is_not_positive")


def test_log_checks_with_custom_check_function(mock_workspace_client: WorkspaceClient):
    spark = _mock_spark()
    engine = DQEngineCore(mock_workspace_client, spark, ExtraParams(checks_log_location=LOG_LOCATION))
    checks = [{"criticality": "error", "check": {"function": "_custom_is_positive", "arguments": {"column": "a"}}}]
    custom_check_functions = {"_custom_is_positive": _custom_is_positive}

    rule_set_fingerprint = engine.log_checks(checks, custom_check_functions=custom_check_functions)

    assert rule_set_fingerprint == compute_rule_set_fingerprint_by_metadata(checks, custom_check_functions)
    assert _logged_rows(spark)[0][2]["function"] == "_custom_is_positive"


def _colliding_rules(name: str = "") -> list[DQRule]:
    return [
        DQRowRule(
            check_func=check_funcs.is_not_null, column=F.try_element_at(F.col("m"), F.lit("k")), name=f"{name}lit"
        ),
        DQRowRule(
            check_func=check_funcs.is_not_null, column=F.try_element_at(F.col("m"), F.col("k")), name=f"{name}col"
        ),
    ]


def test_complex_column_arguments_with_same_fingerprint_raise(mock_workspace_client: WorkspaceClient):
    rules = _colliding_rules()
    # the explicit names differ, so rename both to the same name to force a fingerprint collision
    rules = [rule.replace(name="same") for rule in rules]
    assert rules[0].rule_fingerprint == rules[1].rule_fingerprint
    engine = DQEngineCore(mock_workspace_client, _mock_spark(), ExtraParams(checks_log_location=LOG_LOCATION))

    with pytest.raises(InvalidCheckError, match="distinct 'name'"):
        engine.log_checks(rules)


def test_complex_column_arguments_with_distinct_names_are_logged(
    mock_workspace_client: WorkspaceClient, caplog: pytest.LogCaptureFixture
):
    spark = _mock_spark()
    engine = DQEngineCore(mock_workspace_client, spark, ExtraParams(checks_log_location=LOG_LOCATION))

    engine.log_checks(_colliding_rules("key_"))

    assert len(_logged_rows(spark)) == 2
    assert "display text" in caplog.text


def test_identical_simple_checks_are_not_a_collision(mock_workspace_client: WorkspaceClient):
    spark = _mock_spark()
    engine = DQEngineCore(mock_workspace_client, spark, ExtraParams(checks_log_location=LOG_LOCATION))
    rule: DQRule = DQRowRule(check_func=check_funcs.is_not_null, column="a")

    engine.log_checks([rule, rule])

    assert spark.createDataFrame.call_count == 1


def test_checks_column_is_a_reserved_result_column_when_log_enabled(mock_workspace_client: WorkspaceClient):
    engine = DQEngineCore(mock_workspace_client, _mock_spark(), ExtraParams(checks_log_location=LOG_LOCATION))
    rules: list[DQRule] = [DQRowRule(check_func=check_funcs.is_not_null, column="a")]

    with pytest.raises(InvalidParameterError, match="_checks"):
        engine.apply_checks(_mock_input_df(["a", "_checks"]), rules)


def test_custom_checks_column_name_is_reserved_when_log_enabled(mock_workspace_client: WorkspaceClient):
    extra_params = ExtraParams(checks_log_location=LOG_LOCATION, result_column_names={"checks": "dq_checks"})
    engine = DQEngineCore(mock_workspace_client, _mock_spark(), extra_params)
    rules: list[DQRule] = [DQRowRule(check_func=check_funcs.is_not_null, column="a")]

    with pytest.raises(InvalidParameterError, match="dq_checks"):
        engine.apply_checks(_mock_input_df(["a", "dq_checks"]), rules)


def test_checks_column_is_not_reserved_when_log_disabled(mock_workspace_client: WorkspaceClient):
    engine = DQEngineCore(mock_workspace_client, _mock_spark())
    df = _mock_input_df(["a", "_checks"])

    engine.apply_checks(df, [])

    df.select.assert_called_once()


def test_apply_in_declarative_pipeline_raises_when_checks_log_table_missing(mock_workspace_client: MagicMock):
    spark = _mock_spark(pipeline_id="pipeline-1")
    mock_workspace_client.tables.get.side_effect = NotFound("checks log does not exist yet")
    engine = DQEngineCore(mock_workspace_client, spark, ExtraParams(checks_log_location=LOG_LOCATION))

    with pytest.raises(InvalidConfigError, match="log_checks once outside the pipeline"):
        engine.apply_checks_by_metadata(_mock_input_df(["a", "b"]), METADATA_CHECKS)

    spark.createDataFrame.return_value.write.saveAsTable.assert_not_called()


def test_log_checks_in_declarative_pipeline_appends_to_existing_checks_log(mock_workspace_client: MagicMock):
    spark = _mock_spark(pipeline_id="pipeline-1")
    engine = DQEngineCore(mock_workspace_client, spark, ExtraParams(checks_log_location=LOG_LOCATION))

    engine.log_checks(METADATA_CHECKS, run_config_name="orders")

    spark.createDataFrame.return_value.write.saveAsTable.assert_called_once_with(LOG_LOCATION, mode="append")
    # the existing table is not altered: append-only is only set when the table is created
    assert not any("TBLPROPERTIES" in call.args[0] for call in spark.sql.call_args_list)


def _all_checks_from_resources() -> list[dict]:
    checks: list[dict] = []
    for resource in ("all_row_checks.yaml", "all_dataset_checks.yaml"):
        checks += yaml.safe_load((RESOURCES / resource).read_text(encoding="utf-8"))
    # these checks need a Spark session to build their condition, so they are covered by integration tests
    needs_spark = {"has_valid_schema", "has_valid_json_schema", "is_valid_ipv6_address", "is_ipv6_address_in_cidr"}
    return [check for check in checks if check["check"]["function"] not in needs_spark]


@pytest.mark.parametrize("check", _all_checks_from_resources(), ids=lambda check: check["check"]["function"])
def test_serialized_rules_reproduce_their_fingerprints(check: dict):
    """Rules built in code are logged via serialize_checks; the logged form must reproduce the fingerprints."""
    rules = deserialize_checks([check])
    serialized = ChecksNormalizer.denormalize(serialize_checks(rules))

    assert [rule.rule_fingerprint for rule in deserialize_checks(serialized)] == [
        rule.rule_fingerprint for rule in rules
    ]
    assert compute_rule_set_fingerprint_by_metadata(serialized) == compute_rule_set_fingerprint(rules)
