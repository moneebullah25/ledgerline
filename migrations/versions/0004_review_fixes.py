"""review fixes: eval_run.name (B7), market_price provenance (B11), uq_ingest_failure (B12)

Revision ID: 0004
Revises: 0003
"""
import sqlalchemy as sa
from alembic import op

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None

YAHOO_NOTE = ("Yahoo Finance chart API; personal/research use under Yahoo's terms, "
              "not redistributed")


def upgrade() -> None:
    # B7: existing rows are named after their layer, the same default new writers get.
    op.add_column("eval_run", sa.Column("name", sa.String(length=40), nullable=True))
    op.execute("UPDATE eval_run SET name = layer")
    op.alter_column("eval_run", "name", nullable=False)
    op.create_index("ix_eval_run_latest", "eval_run", ["layer", "name", "created_at"])

    # B11: rows loaded before 0004 never hashed their response; say so instead of inventing one.
    op.add_column("market_price", sa.Column("sha256", sa.String(length=64), nullable=True))
    op.add_column("market_price", sa.Column("license_note", sa.Text(), nullable=True))
    op.execute(sa.text("UPDATE market_price SET sha256 = 'unrecorded', license_note = :n")
               .bindparams(n=YAHOO_NOTE))
    op.alter_column("market_price", "sha256", nullable=False)
    op.alter_column("market_price", "license_note", nullable=False)

    # B12: keep the oldest row of any duplicate before the constraint can exist.
    op.execute("""
        DELETE FROM ingest_failure a USING ingest_failure b
        WHERE a.flow = b.flow AND a.key = b.key AND a.error_class = b.error_class AND a.id > b.id
    """)
    op.create_unique_constraint("uq_ingest_failure", "ingest_failure",
                                ["flow", "key", "error_class"])


def downgrade() -> None:
    op.drop_constraint("uq_ingest_failure", "ingest_failure", type_="unique")
    op.drop_column("market_price", "license_note")
    op.drop_column("market_price", "sha256")
    op.drop_index("ix_eval_run_latest", table_name="eval_run")
    op.drop_column("eval_run", "name")
