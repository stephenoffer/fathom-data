"""The last hop: where a model meets a user.

The rest of `ai` stops at the model. Everything between a model and a request —
which variant is behind the endpoint, what share of traffic it takes, whether it was
quantized on the way — changes what people actually see and is invisible to lineage.

    deployments   endpoints, variants, traffic splits, rollout and rollback

The endpoint is a dataset like everything else, so "which model answered production
traffic on Tuesday" is a graph query rather than a spelunk through deploy logs.
"""

from . import deployments
from .deployments import (
    CapabilityResult,
    Deployment,
    DeploymentState,
    RegressionReport,
    RolloutStrategy,
    TrafficSplit,
    Variant,
    active_variants,
    can_rollback,
    deployment_edges,
    is_canary,
    promote,
    regression_report,
    rollback,
    rollout_plan,
    traffic_to,
    validate_split,
)

__all__ = [
    "CapabilityResult",
    "Deployment",
    "DeploymentState",
    "RegressionReport",
    "RolloutStrategy",
    "TrafficSplit",
    "Variant",
    "active_variants",
    "can_rollback",
    "deployment_edges",
    "deployments",
    "is_canary",
    "promote",
    "regression_report",
    "rollback",
    "rollout_plan",
    "traffic_to",
    "validate_split",
]
