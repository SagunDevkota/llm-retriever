"""Phase 4 — save checkpoint to Postgres.
    python save_checkpoint.py --db <database_name>

The database name is required and is swapped into POSTGRES_DSN (which supplies
host / user / port); this way loading data always names its target database
explicitly instead of silently defaulting.
"""

import argparse

from core.checkpoint import load_checkpoint
from core.config import CHECKPOINT_FILE, POSTGRES_DSN, dsn_with_db
from db.db2 import PostgresStore


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Phase 4: bulk-load the checkpoint into a Postgres database."
    )
    ap.add_argument(
        "--db", required=True,
        help="Target database name (required). Host/user/port come from POSTGRES_DSN.",
    )
    args = ap.parse_args()

    dsn = dsn_with_db(POSTGRES_DSN, args.db)

    nodes_by_id = load_checkpoint(CHECKPOINT_FILE)   # reads hierarchical_checkpoint.json
    if not nodes_by_id:
        raise SystemExit("No checkpoint found. Run run_parse.py + run_summarize.py first.")
    print(f"{len(nodes_by_id)} nodes loaded")

    ps = PostgresStore(dsn)                # bulk upsert of evidence_/routing_ columns
    ps.insert_nodes_bulk(nodes_by_id)
    ps.close()
    print("done")
