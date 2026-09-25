"""platform round 3: verdict(claim_id, id) index, evidence.cache_write_tokens (K5)

Revision ID: 0005
Revises: 0004
"""
import sqlalchemy as sa
from alembic import op

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_index("ix_verdict_claim", "verdict", ["claim_id", "id"])
    # Rows written before 0005 did not keep the token count (their cost_usd already priced
    # the writes), so they read 0.
    op.add_column("evidence", sa.Column("cache_write_tokens", sa.Integer(),
                                        server_default="0", nullable=False))


def downgrade() -> None:
    op.drop_column("evidence", "cache_write_tokens")
    op.drop_index("ix_verdict_claim", table_name="verdict")
