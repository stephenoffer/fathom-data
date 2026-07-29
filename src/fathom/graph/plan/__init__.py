"""What a plan costs, and how it turns into work.

Separate from the planner because these answer different questions with different
failure modes. The planner must never under-invalidate. These may be wrong by
twenty per cent and still do their job, which is to make the saving legible and the
rebuild runnable.

    billing   what the warehouse actually charged, against what the model predicted
    cost      what this run costs, and what skipping the rest saved
    lifetime  what a dataset has cost since it existed, against whether anyone reads it
    schedule  the plan arranged into waves an orchestrator can run
"""

from . import billing, cost, lifetime, schedule

__all__ = ["billing", "cost", "lifetime", "schedule"]
