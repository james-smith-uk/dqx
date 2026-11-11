import pytest
from pyspark.errors import AnalysisException

from databricks.labs.dqx import check_funcs
from databricks.labs.dqx.config import ExtraParams
from databricks.labs.dqx.engine import DQEngine
from databricks.labs.dqx.errors import UnsafeSqlQueryError
from databricks.labs.dqx.rule import DQDatasetRule, DQRowRule
from tests.integration.conftest import RUN_ID, RUN_TIME

SCHEMA = "a: int, b: int, c: int"


@pytest.fixture
def engine_without_skipping(ws):
    extra_params = ExtraParams(
        run_time_overwrite=RUN_TIME.isoformat(), run_id_overwrite=RUN_ID, skip_invalid_checks=False
    )
    return DQEngine(workspace_client=ws, extra_params=extra_params)


def test_apply_checks_without_skipping_matches_default_results_for_valid_checks(ws, spark, engine_without_skipping):
    default_engine = DQEngine(
        workspace_client=ws, extra_params=ExtraParams(run_time_overwrite=RUN_TIME.isoformat(), run_id_overwrite=RUN_ID)
    )
    test_df = spark.createDataFrame([[1, None, 3], [None, 2, 3]], SCHEMA)
    checks = [
        DQRowRule(name="a_is_null", criticality="error", check_func=check_funcs.is_not_null, column="a"),
        DQRowRule(
            name="b_is_null_when_c",
            criticality="warn",
            check_func=check_funcs.is_not_null,
            column="b",
            filter="c > 0",
        ),
    ]

    expected = default_engine.apply_checks(test_df, checks).orderBy("c", "a").collect()
    actual = engine_without_skipping.apply_checks(test_df, checks).orderBy("c", "a").collect()

    assert actual == expected


@pytest.mark.parametrize(
    "check",
    [
        DQRowRule(name="missing_col", criticality="error", check_func=check_funcs.is_not_null, column="missing_col"),
        DQRowRule(
            name="missing_filter_col",
            criticality="error",
            check_func=check_funcs.is_not_null,
            column="a",
            filter="missing_col > 0",
        ),
        DQDatasetRule(
            name="missing_unique_col", criticality="error", check_func=check_funcs.is_unique, columns=["missing_col"]
        ),
    ],
    ids=["row_check_column", "filter", "dataset_check_column"],
)
def test_apply_checks_without_skipping_lets_spark_reject_invalid_check(spark, engine_without_skipping, check):
    test_df = spark.createDataFrame([[1, 2, 3]], SCHEMA)

    with pytest.raises(AnalysisException):
        engine_without_skipping.apply_checks(test_df, [check]).collect()


def test_apply_checks_without_skipping_raises_for_unsafe_filter(spark, engine_without_skipping):
    test_df = spark.createDataFrame([[1, 2, 3]], SCHEMA)
    check = DQRowRule(
        name="unsafe_filter",
        criticality="error",
        check_func=check_funcs.is_not_null,
        column="a",
        filter="1=1; DROP TABLE t",
    )

    with pytest.raises(UnsafeSqlQueryError):
        engine_without_skipping.apply_checks(test_df, [check])
