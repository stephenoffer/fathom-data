"""Project configuration parsing.

Most of these are about failing loudly. A config that silently ignores a misspelled
key produces a project that looks configured and plans wrong, which is strictly
worse than one that refuses to load.
"""

from __future__ import annotations

import pytest

from fathom.cli.config import find_config, load_config, parse_config
from fathom.core.errors import ConfigError
from fathom.core.grains import Grain
from fathom.core.ids import normalize_table


def write(tmp_path, body: str, name: str = "fathom.yml"):
    (tmp_path / name).write_text(body)
    return tmp_path / name


MINIMAL = """
version: 1
system: duckdb
datasets:
  - name: raw.events
    partition: [{field: dt, grain: day}, {field: region}]
"""


# -- basics --------------------------------------------------------------------


def test_loads_a_minimal_config(tmp_path):
    config = load_config(write(tmp_path, MINIMAL))
    assert config.system == "duckdb"
    assert len(config.datasets) == 1
    spec = config.datasets[0].spec
    assert spec.field("dt").grain is Grain.DAY
    assert spec.field("region").kind == "value"


def test_store_path_is_relative_to_the_config_not_the_cwd(tmp_path):
    """The same project must work from any working directory."""
    config = load_config(write(tmp_path, MINIMAL + "\nstore: state/db.sqlite\n"))
    assert config.store == tmp_path / "state" / "db.sqlite"


def test_model_paths_are_relative_to_the_config(tmp_path):
    body = (
        MINIMAL
        + """
  - name: gold.monthly
    model: models/gold.sql
"""
    )
    config = load_config(write(tmp_path, body))
    entry = config.dataset("gold.monthly")
    assert entry is not None and entry.sql_path == tmp_path / "models" / "gold.sql"
    assert entry.is_model


def test_config_is_found_by_searching_upward(tmp_path):
    write(tmp_path, MINIMAL)
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)
    assert find_config(nested) == tmp_path / "fathom.yml"


def test_an_explicitly_named_missing_config_says_so(tmp_path):
    with pytest.raises(ConfigError, match="does not exist"):
        load_config(tmp_path / "nope.yml")


def test_searching_and_finding_nothing_says_what_to_create(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ConfigError, match="fathom.yml"):
        load_config()


# -- validation ----------------------------------------------------------------


def test_unknown_top_level_keys_are_rejected(tmp_path):
    """A silently ignored typo produces a config that looks right and plans wrong."""
    with pytest.raises(ConfigError, match="unknown key"):
        load_config(write(tmp_path, MINIMAL + "\ndatsets: []\n"))


def test_unknown_dataset_keys_are_rejected(tmp_path):
    body = """
version: 1
datasets:
  - name: raw.events
    partiton: [dt]
"""
    with pytest.raises(ConfigError, match="unknown key"):
        load_config(write(tmp_path, body))


def test_error_messages_list_the_valid_keys(tmp_path):
    with pytest.raises(ConfigError) as caught:
        load_config(write(tmp_path, MINIMAL + "\nnonsense: 1\n"))
    assert "Valid keys" in str(caught.value)


def test_unsupported_version_is_rejected(tmp_path):
    with pytest.raises(ConfigError, match="version 2 is not supported"):
        load_config(write(tmp_path, "version: 2\n"))


def test_duplicate_datasets_are_rejected(tmp_path):
    body = """
version: 1
datasets:
  - name: raw.events
  - name: raw.events
"""
    with pytest.raises(ConfigError, match="declared twice"):
        load_config(write(tmp_path, body))


def test_dataset_without_a_name_is_rejected(tmp_path):
    with pytest.raises(ConfigError, match="needs a `name`"):
        load_config(write(tmp_path, "version: 1\ndatasets:\n  - partition: [dt]\n"))


def test_unknown_grain_is_rejected(tmp_path):
    body = """
version: 1
datasets:
  - name: raw.events
    partition: [{field: dt, grain: fortnight}]
"""
    with pytest.raises(ConfigError, match="unknown grain"):
        load_config(write(tmp_path, body))


def test_invalid_lineage_type_lists_the_options(tmp_path):
    body = MINIMAL + "\nlineage:\n  - type: telepathy\n"
    with pytest.raises(ConfigError, match="sql, dbt, openlineage, adapter"):
        load_config(write(tmp_path, body))


def test_malformed_yaml_is_rejected_clearly(tmp_path):
    with pytest.raises(ConfigError, match="not valid YAML"):
        load_config(write(tmp_path, "datasets: [\n  - broken"))


# -- environment references ----------------------------------------------------


def test_env_references_are_expanded(tmp_path, monkeypatch):
    monkeypatch.setenv("FATHOM_TEST_ACCOUNT", "xy12345")
    config = load_config(write(tmp_path, MINIMAL + '\ninstance: "${FATHOM_TEST_ACCOUNT}"\n'))
    assert config.instance == "xy12345"


def test_env_references_support_defaults(tmp_path, monkeypatch):
    monkeypatch.delenv("FATHOM_TEST_MISSING", raising=False)
    config = load_config(
        write(tmp_path, MINIMAL + '\ninstance: "${FATHOM_TEST_MISSING:-fallback}"\n')
    )
    assert config.instance == "fallback"


def test_missing_env_reference_says_which_variable(tmp_path, monkeypatch):
    """Secrets belong in the environment, so the failure must name the variable."""
    monkeypatch.delenv("FATHOM_TEST_ABSENT", raising=False)
    with pytest.raises(ConfigError, match="FATHOM_TEST_ABSENT"):
        load_config(write(tmp_path, MINIMAL + '\ninstance: "${FATHOM_TEST_ABSENT}"\n'))


def test_env_references_expand_inside_nested_options(tmp_path, monkeypatch):
    monkeypatch.setenv("FATHOM_TEST_KEY", "secret-value")
    body = MINIMAL + '\nstorage_options:\n  s3: {key: "${FATHOM_TEST_KEY}"}\n'
    config = load_config(write(tmp_path, body))
    assert config.options_for("s3")["key"] == "secret-value"


# -- resolution ----------------------------------------------------------------


def test_bare_names_resolve_through_the_system(tmp_path):
    config = load_config(write(tmp_path, MINIMAL.replace("duckdb", "snowflake")))
    assert config.datasets[0].dataset == normalize_table("raw.events", system="snowflake")


def test_uris_resolve_as_paths_and_round_trip(tmp_path):
    """`Path("file:///a")` collapses the slashes, so URIs must bypass `Path`."""
    config = load_config(write(tmp_path, MINIMAL))
    dataset = config.resolve("s3://lake/raw/events")
    assert str(dataset) == "s3://lake/raw/events"
    assert config.resolve(str(dataset)) == dataset


def test_resolved_identities_look_up_the_same_entry(tmp_path):
    body = """
version: 1
datasets:
  - name: /tmp/lake/events
    adapter: storage
"""
    config = load_config(write(tmp_path, body))
    dataset = config.datasets[0].dataset
    assert config.dataset(dataset) is config.datasets[0]


def test_relative_paths_resolve_against_the_config_directory(tmp_path):
    (tmp_path / "data").mkdir()
    config = load_config(write(tmp_path, "version: 1\ndatasets:\n  - name: ./data\n"))
    assert str(tmp_path / "data") in str(config.datasets[0].dataset)


# -- policies ------------------------------------------------------------------


def test_policies_parse(tmp_path):
    body = (
        MINIMAL
        + """
policies:
  - dataset: ml.training_set
    forbid: [pii, national_id]
    reason: not cleared
"""
    )
    config = load_config(write(tmp_path, body))
    policy = config.policies[0]
    assert policy.forbid == frozenset({"pii", "national_id"})
    assert policy.reason == "not cleared"


def test_policy_without_a_dataset_is_rejected(tmp_path):
    with pytest.raises(ConfigError, match="missing `dataset`"):
        load_config(write(tmp_path, MINIMAL + "\npolicies:\n  - forbid: [pii]\n"))


# -- direct parsing ------------------------------------------------------------


def test_parse_config_accepts_a_mapping(tmp_path):
    config = parse_config({"version": 1, "system": "trino"}, root=tmp_path)
    assert config.system == "trino"


def test_non_mapping_config_is_rejected(tmp_path):
    with pytest.raises(ConfigError, match="mapping at the top level"):
        parse_config(["not", "a", "mapping"], root=tmp_path)  # type: ignore[arg-type]


# -- messages that teach the shape ---------------------------------------------
#
# A config file is the first thing a new user writes and the thing they have least
# to go on. Each of these asserts the message shows the YAML that would have worked,
# because "invalid partition entry" sends the reader back to the documentation and a
# worked example does not.


def test_an_unknown_key_suggests_the_one_that_was_meant(tmp_path):
    path = write(tmp_path, "version: 1\nsystm: duckdb\n")
    with pytest.raises(ConfigError) as exc:
        load_config(path)
    message = str(exc.value)
    assert "Did you mean 'system'?" in message
    assert "Valid keys here:" in message


def test_a_bare_string_partition_is_accepted_as_a_single_value_field(tmp_path):
    """`partition: region` is unambiguous, so making the reader bracket it is noise."""
    path = write(tmp_path, "version: 1\ndatasets:\n  - name: raw.events\n    partition: region\n")
    assert load_config(path).datasets[0].spec.names == ("region",)


def test_a_partition_that_is_not_a_list_shows_one_that_is(tmp_path):
    path = write(tmp_path, "version: 1\ndatasets:\n  - name: raw.events\n    partition: 3\n")
    with pytest.raises(ConfigError) as exc:
        load_config(path)
    assert "{field: dt, grain: day}" in str(exc.value)


def test_a_partition_entry_missing_field_names_the_key(tmp_path):
    path = write(
        tmp_path,
        "version: 1\ndatasets:\n  - name: raw.events\n    partition:\n      - {grain: day}\n",
    )
    with pytest.raises(ConfigError) as exc:
        load_config(path)
    assert "needs a `field`" in str(exc.value)


def test_an_unknown_grain_suggests_a_real_one(tmp_path):
    path = write(
        tmp_path,
        "version: 1\ndatasets:\n  - name: raw.events\n"
        "    partition:\n      - {field: dt, grain: daly}\n",
    )
    with pytest.raises(ConfigError) as exc:
        load_config(path)
    assert "Did you mean 'day'?" in str(exc.value)


def test_a_grain_written_as_an_adjective_is_accepted(tmp_path):
    """Config files write `day`, cron descriptions write `daily`. Both mean one thing."""
    path = write(
        tmp_path,
        "version: 1\ndatasets:\n  - name: raw.events\n"
        "    partition:\n      - {field: dt, grain: daily}\n",
    )
    config = load_config(path)
    assert config.datasets[0].spec.field("dt").grain is Grain.DAY


def test_a_top_level_list_says_how_to_see_the_right_shape(tmp_path):
    path = write(tmp_path, "- version: 1\n")
    with pytest.raises(ConfigError) as exc:
        load_config(path)
    message = str(exc.value)
    assert "stray leading `-`" in message
    assert "fathom init" in message


def test_a_duplicate_dataset_explains_why_it_cannot_be_merged(tmp_path):
    path = write(
        tmp_path,
        "version: 1\ndatasets:\n  - name: raw.events\n  - name: raw.events\n",
    )
    with pytest.raises(ConfigError) as exc:
        load_config(path)
    assert "nothing can arbitrate between them" in str(exc.value)


def test_an_unsupported_version_offers_the_likely_cause(tmp_path):
    path = write(tmp_path, "version: 2\n")
    with pytest.raises(ConfigError) as exc:
        load_config(path)
    assert "typo" in str(exc.value)
