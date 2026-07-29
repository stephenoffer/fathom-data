"""Storage: objects, prefixes, and the weakest change detection on the ladder.

One implementation covers S3, GCS, ADLS, R2, MinIO, HDFS, and local disk, because
the difference between a bucket and a directory is fsspec's problem rather than the
planner's. It reports `LIST_DIFF` on purpose: if everything downstream works on a
LIST plus etag comparison, a catalog adapter with real snapshot diffs is strictly
faster rather than differently shaped.
"""

from .objects import LocalStorage, ObjectStorage

__all__ = ["LocalStorage", "ObjectStorage"]
