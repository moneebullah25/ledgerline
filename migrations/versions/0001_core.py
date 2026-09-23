"""core: companies, filings, XBRL facts, prices, ingest failures

Revision ID: 0001
Revises: None
"""
from alembic import op
import sqlalchemy as sa
import pgvector.sqlalchemy
from sqlalchemy.dialects import postgresql

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table('company',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('cik', sa.String(length=10), nullable=False),
    sa.Column('ticker', sa.String(length=10), nullable=False),
    sa.Column('name', sa.Text(), nullable=False),
    sa.Column('fiscal_year_end', sa.String(length=4), nullable=True),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('cik'),
    sa.UniqueConstraint('ticker')
    )
    op.create_table('ingest_failure',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('flow', sa.String(length=40), nullable=False),
    sa.Column('key', sa.Text(), nullable=False),
    sa.Column('error_class', sa.String(length=60), nullable=False),
    sa.Column('error', sa.Text(), nullable=False),
    sa.Column('payload', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('market_price',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('ticker', sa.String(length=10), nullable=False),
    sa.Column('day', sa.Date(), nullable=False),
    sa.Column('close', sa.Float(), nullable=False),
    sa.Column('adj_close', sa.Float(), nullable=False),
    sa.Column('volume', sa.Numeric(precision=20, scale=0), nullable=True),
    sa.Column('source_url', sa.Text(), nullable=False),
    sa.Column('fetched_at', sa.DateTime(timezone=True), nullable=False),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('ticker', 'day', name='uq_price')
    )
    op.create_table('filing',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('company_id', sa.Integer(), nullable=False),
    sa.Column('accession_no', sa.String(length=20), nullable=False),
    sa.Column('form_type', sa.String(length=10), nullable=False),
    sa.Column('filed_at', sa.Date(), nullable=False),
    sa.Column('period_of_report', sa.Date(), nullable=True),
    sa.Column('fiscal_year', sa.Integer(), nullable=True),
    sa.Column('fiscal_period', sa.String(length=2), nullable=True),
    sa.Column('doc_url', sa.Text(), nullable=False),
    sa.Column('storage_path', sa.Text(), nullable=False),
    sa.Column('source_url', sa.Text(), nullable=False),
    sa.Column('fetched_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('sha256', sa.String(length=64), nullable=False),
    sa.Column('license_note', sa.Text(), nullable=False),
    sa.CheckConstraint("form_type IN ('10-K','10-Q','8-K','10-K/A','10-Q/A')", name='form'),
    sa.ForeignKeyConstraint(['company_id'], ['company.id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('accession_no')
    )
    op.create_index(op.f('ix_filing_company_id'), 'filing', ['company_id'], unique=False)
    op.create_table('xbrl_fact',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('company_id', sa.Integer(), nullable=False),
    sa.Column('tag', sa.Text(), nullable=False),
    sa.Column('unit', sa.String(length=20), nullable=False),
    sa.Column('value', sa.Numeric(precision=28, scale=6), nullable=False),
    sa.Column('period_start', sa.Date(), nullable=True),
    sa.Column('period_end', sa.Date(), nullable=False),
    sa.Column('duration_days', sa.Integer(), nullable=True),
    sa.Column('fiscal_year', sa.Integer(), nullable=False),
    sa.Column('fiscal_period', sa.String(length=2), nullable=False),
    sa.Column('form', sa.String(length=10), nullable=False),
    sa.Column('accession_no', sa.String(length=20), nullable=False),
    sa.Column('filed', sa.Date(), nullable=False),
    sa.Column('derived', sa.Boolean(), nullable=False),
    sa.Column('source_url', sa.Text(), nullable=False),
    sa.Column('fetched_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('sha256', sa.String(length=64), nullable=False),
    sa.Column('license_note', sa.Text(), nullable=False),
    sa.CheckConstraint("fiscal_period IN ('Q1', 'Q2', 'Q3', 'Q4', 'FY')", name='fp'),
    sa.ForeignKeyConstraint(['company_id'], ['company.id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('company_id', 'tag', 'unit', 'period_start', 'period_end', 'accession_no', name='uq_xbrl_fact', postgresql_nulls_not_distinct=True)
    )
    op.create_index('ix_xbrl_lookup', 'xbrl_fact', ['company_id', 'tag', 'unit', 'fiscal_year', 'fiscal_period'], unique=False)

def downgrade() -> None:
    op.drop_index('ix_xbrl_lookup', table_name='xbrl_fact')
    op.drop_table('xbrl_fact')
    op.drop_index(op.f('ix_filing_company_id'), table_name='filing')
    op.drop_table('filing')
    op.drop_table('market_price')
    op.drop_table('ingest_failure')
    op.drop_table('company')
