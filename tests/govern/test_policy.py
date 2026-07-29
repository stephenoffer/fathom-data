"""Label inference, propagation, and enforcement."""

from __future__ import annotations

from fathom.core.ids import normalize_table
from fathom.core.partitions import PartitionMapping
from fathom.core.types import UNPARTITIONED, ColumnRef
from fathom.govern.policy import (
    UNATTRIBUTED,
    Label,
    SinkPolicy,
    enforce,
    infer,
    propagate,
)
from fathom.graph import Edge, Graph
from fathom.observe.profile import ColumnProfile, Profile

RAW = normalize_table("raw.users", system="duckdb")
SILVER = normalize_table("silver.users", system="duckdb")
TRAINING = normalize_table("ml.training_set", system="duckdb")


def profile(*columns: ColumnProfile) -> Profile:
    return Profile(dataset=RAW, row_count=100, columns=columns)


def names(labels, ref) -> set[str]:
    return {label.name for label in labels.get(ref, set())}


# -- inference -----------------------------------------------------------------


def test_name_and_type_propose_a_label():
    got = infer(profile(ColumnProfile(name="email_address", dtype="string", row_count=100)))
    assert "email" in names(got, ColumnRef(RAW, "email_address"))


def test_personal_labels_imply_pii():
    got = infer(profile(ColumnProfile(name="date_of_birth", dtype="date", row_count=100)))
    assert "pii" in names(got, ColumnRef(RAW, "date_of_birth"))


def test_non_personal_labels_do_not_imply_pii():
    got = infer(profile(ColumnProfile(name="total_amount", dtype="double", row_count=100)))
    labels = names(got, ColumnRef(RAW, "total_amount"))
    assert "monetary_amount" in labels
    assert "pii" not in labels


def test_type_mismatch_blocks_a_name_match():
    """A column called `email` holding doubles is not an email address."""
    got = infer(profile(ColumnProfile(name="email", dtype="double", row_count=100)))
    assert "email" not in names(got, ColumnRef(RAW, "email"))


def test_statistics_corroborate_and_raise_confidence():
    got = infer(
        profile(ColumnProfile(name="latitude", dtype="double", row_count=100, min=-33.9, max=51.5))
    )
    label = next(x for x in got[ColumnRef(RAW, "latitude")] if x.name == "latitude")
    assert label.confidence > 0.7
    assert label.origin == "inferred:name+stats"


def test_statistics_refute_an_impossible_range():
    """Footer stats are the cheapest possible check against a misleading name."""
    got = infer(
        profile(ColumnProfile(name="latitude", dtype="double", row_count=100, min=0.0, max=4000.0))
    )
    assert ColumnRef(RAW, "latitude") not in got


def test_columns_with_no_signal_are_left_alone():
    got = infer(profile(ColumnProfile(name="col_a", dtype="string", row_count=100)))
    assert got == {}


def test_inference_never_claims_certainty():
    """These are proposals for a human to confirm, not conclusions."""
    got = infer(profile(ColumnProfile(name="ssn", dtype="string", row_count=100)))
    assert all(
        label.confidence < 1.0 and not label.confirmed for label in got[ColumnRef(RAW, "ssn")]
    )


# -- propagation ---------------------------------------------------------------


def linked_graph(columns=(("email_address", "contact"),)) -> Graph:
    g = Graph()
    for ds in (RAW, SILVER, TRAINING):
        g.add_dataset(ds, UNPARTITIONED)
    g.add_edge(Edge(RAW, SILVER, PartitionMapping(), columns=tuple(columns)))
    g.add_edge(Edge(SILVER, TRAINING, PartitionMapping(), columns=(("contact", "feature_1"),)))
    return g


def test_labels_flow_downstream_along_column_edges():
    seeds = {ColumnRef(RAW, "email_address"): {Label("pii", 0.9, "inferred")}}
    got = propagate(linked_graph(), seeds)
    assert "pii" in names(got, ColumnRef(SILVER, "contact"))
    assert "pii" in names(got, ColumnRef(TRAINING, "feature_1"))


def test_confidence_decays_with_distance():
    seeds = {ColumnRef(RAW, "email_address"): {Label("pii", 0.9, "inferred")}}
    got = propagate(linked_graph(), seeds)
    near = next(x for x in got[ColumnRef(SILVER, "contact")] if x.name == "pii")
    far = next(x for x in got[ColumnRef(TRAINING, "feature_1")] if x.name == "pii")
    assert 0.9 > near.confidence > far.confidence


def test_edges_without_column_lineage_still_carry_the_label():
    """Losing track of PII because a job used the DataFrame API is the worst outcome."""
    g = Graph()
    g.add_dataset(RAW, UNPARTITIONED)
    g.add_dataset(SILVER, UNPARTITIONED)
    g.add_edge(Edge(RAW, SILVER, PartitionMapping(), columns=(), evidence="spark-rdd"))

    got = propagate(g, {ColumnRef(RAW, "ssn"): {Label("pii", 0.9, "inferred")}})
    assert "pii" in names(got, ColumnRef(SILVER, UNATTRIBUTED))


def test_propagation_terminates_on_a_cycle():
    g = Graph()
    g.add_dataset(RAW, UNPARTITIONED)
    g.add_dataset(SILVER, UNPARTITIONED)
    g.add_edge(Edge(RAW, SILVER, PartitionMapping(), columns=(("a", "b"),)))
    g.add_edge(Edge(SILVER, RAW, PartitionMapping(), columns=(("b", "a"),)))

    got = propagate(g, {ColumnRef(RAW, "a"): {Label("pii", 0.9, "inferred")}})
    assert "pii" in names(got, ColumnRef(SILVER, "b"))


def test_confirmed_labels_are_not_downgraded_by_propagation():
    seeds = {
        ColumnRef(RAW, "email_address"): {Label("pii", 0.9, "inferred")},
        ColumnRef(SILVER, "contact"): {Label("pii", 0.2, "declared", confirmed=True)},
    }
    got = propagate(linked_graph(), seeds)
    label = next(x for x in got[ColumnRef(SILVER, "contact")] if x.name == "pii")
    assert label.confirmed and label.confidence == 0.2


# -- enforcement ---------------------------------------------------------------


def test_forbidden_label_reaching_a_sink_is_a_violation():
    labels = {ColumnRef(TRAINING, "feature_1"): {Label("pii", 0.8, "propagated")}}
    report = enforce(labels, [SinkPolicy.no_pii(TRAINING)])
    assert not report.ok
    assert report.violations[0].label == "pii"
    assert "not cleared" in report.violations[0].reason


def test_weak_inferences_do_not_block_a_pipeline():
    labels = {ColumnRef(TRAINING, "feature_1"): {Label("pii", 0.2, "propagated")}}
    assert enforce(labels, [SinkPolicy.no_pii(TRAINING)]).ok


def test_a_confirmed_label_always_counts():
    """Confidence is irrelevant once a human has decided."""
    labels = {ColumnRef(TRAINING, "f"): {Label("pii", 0.01, "declared", confirmed=True)}}
    assert not enforce(labels, [SinkPolicy.no_pii(TRAINING)]).ok


def test_missing_required_label_is_a_violation():
    policy = SinkPolicy(dataset=TRAINING, require=frozenset({"consent:training"}))
    report = enforce({ColumnRef(TRAINING, "f"): {Label("pii", 0.9)}}, [policy])
    assert [v.rule for v in report.violations] == ["required label missing"]


def test_datasets_without_a_policy_are_ignored():
    labels = {ColumnRef(SILVER, "contact"): {Label("pii", 0.9, "propagated")}}
    assert enforce(labels, [SinkPolicy.no_pii(TRAINING)]).ok


def test_unattributed_violations_say_the_column_is_unknown():
    labels = {ColumnRef(TRAINING, UNATTRIBUTED): {Label("pii", 0.9, "propagated")}}
    report = enforce(labels, [SinkPolicy.no_pii(TRAINING)])
    assert report.violations[0].is_unattributed
    assert "column unknown" in str(report.violations[0])


def test_end_to_end_pii_reaches_a_training_set():
    """The scenario the whole verb exists for."""
    seeds = infer(profile(ColumnProfile(name="email_address", dtype="string", row_count=100)))
    labels = propagate(linked_graph(), seeds)
    report = enforce(labels, [SinkPolicy.no_pii(TRAINING)])

    assert not report.ok
    assert any(v.dataset == TRAINING and v.label == "pii" for v in report.violations)


def test_email_address_is_not_a_postal_address():
    """`_address` as a suffix is not enough; postal detection needs a postal token."""
    got = infer(profile(ColumnProfile(name="email_address", dtype="string", row_count=100)))
    labels = names(got, ColumnRef(RAW, "email_address"))
    assert "email" in labels
    assert "postal_address" not in labels


def test_ip_address_is_not_a_postal_address():
    got = infer(profile(ColumnProfile(name="ip_address", dtype="string", row_count=100)))
    assert "postal_address" not in names(got, ColumnRef(RAW, "ip_address"))


def test_real_postal_columns_still_match():
    for name in ("address", "street_name", "postcode", "shipping_address"):
        got = infer(profile(ColumnProfile(name=name, dtype="string", row_count=100)))
        assert "postal_address" in names(got, ColumnRef(RAW, name)), name


def test_labels_over_groups_by_label_name():
    """One shared lookup behind three different notions of `reach`."""
    from fathom.govern.policy import labels_over

    a, b, c = RAW, SILVER, TRAINING
    labels = {
        ColumnRef(a, "email"): {Label("pii"), Label("email")},
        ColumnRef(b, "phone"): {Label("pii")},
        ColumnRef(c, "amount"): {Label("monetary_amount")},
    }
    found = labels_over(labels, [a, b])

    assert found["pii"] == sorted([ColumnRef(a, "email"), ColumnRef(b, "phone")], key=str)
    assert found["email"] == [ColumnRef(a, "email")]
    assert "monetary_amount" not in found  # c was out of reach


def test_tag_index_flattens_to_what_selection_needs():
    from fathom.govern.policy import tag_index

    a = RAW
    index = tag_index({ColumnRef(a, "email"): {Label("pii"), Label("email")}})
    assert index == {a: {"pii", "email"}}


def test_an_unattributed_label_keeps_travelling_across_column_level_edges():
    """PII entering via a DataFrame job must not vanish at the next SQL model.

    Column detail on an edge says which columns map where. It does not say the
    unattributed label is absent, so stopping there dropped the label entirely —
    and a `forbid: [pii]` policy on the far side then passed.
    """
    raw = RAW
    mid = SILVER  # reached with no column lineage
    gold = normalize_table("gold.report", system="duckdb")  # reached with column lineage

    g = Graph()
    g.add_edge(Edge(raw, mid, PartitionMapping()))
    g.add_edge(Edge(mid, gold, PartitionMapping(), columns=(("a", "a"), ("b", "b"))))

    out = propagate(g, {ColumnRef(raw, "email"): {Label("pii", 0.9, "inferred")}})

    reached = {ref.dataset for ref, values in out.items() for lab in values if lab.name == "pii"}
    assert gold in reached

    report = enforce(out, [SinkPolicy.no_pii(gold)])
    assert not report.ok


def test_a_customer_id_counts_as_personal_data():
    """It singles a person out, and it is the column `erase` keys on."""
    found = infer(profile(ColumnProfile("user_id", "string", 100)))
    names = {lab.name for labels in found.values() for lab in labels}
    assert "user_identifier" in names
    assert "pii" in names


def test_two_policies_on_one_sink_are_both_enforced():
    """Keeping only the last silently drops half the rules the user wrote."""
    ds = TRAINING
    labels = {
        ColumnRef(ds, "email"): {Label("pii", 0.9, "inferred")},
        ColumnRef(ds, "lic"): {Label("restricted_licence", 0.9, "inferred")},
    }
    report = enforce(
        labels,
        [
            SinkPolicy(dataset=ds, forbid=frozenset({"pii"})),
            SinkPolicy(dataset=ds, forbid=frozenset({"restricted_licence"})),
        ],
    )
    assert {v.label for v in report.violations} == {"pii", "restricted_licence"}
