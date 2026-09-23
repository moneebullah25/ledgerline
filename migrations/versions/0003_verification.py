"""verification: claims, reports, evidence, verdicts, traces, eval runs

Revision ID: 0003
Revises: 0002
"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0003"
down_revision = '0002'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table('eval_run',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('git_sha', sa.String(length=40), nullable=True),
    sa.Column('layer', sa.String(length=4), nullable=False),
    sa.Column('metrics', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.Column('details', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('span',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('trace_id', sa.UUID(), nullable=False),
    sa.Column('parent_id', sa.UUID(), nullable=True),
    sa.Column('name', sa.Text(), nullable=False),
    sa.Column('started_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('ended_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('attrs', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_span_trace_id'), 'span', ['trace_id'], unique=False)
    op.create_table('report',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('company_id', sa.Integer(), nullable=False),
    sa.Column('fiscal_year', sa.Integer(), nullable=False),
    sa.Column('fiscal_period', sa.String(length=2), nullable=False),
    sa.Column('status', sa.String(length=10), nullable=False),
    sa.Column('config', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('finished_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('latency_s', sa.Float(), nullable=True),
    sa.Column('cost_usd', sa.Float(), nullable=True),
    sa.Column('error', sa.Text(), nullable=True),
    sa.CheckConstraint("status IN ('queued','running','done','failed')", name='status'),
    sa.ForeignKeyConstraint(['company_id'], ['company.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_report_created', 'report', ['created_at', 'id'], unique=False)
    op.create_table('claim',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('company_id', sa.Integer(), nullable=False),
    sa.Column('fiscal_year', sa.Integer(), nullable=False),
    sa.Column('fiscal_period', sa.String(length=2), nullable=False),
    sa.Column('filing_id', sa.Integer(), nullable=True),
    sa.Column('segment_id', sa.Integer(), nullable=True),
    sa.Column('span_start', sa.Integer(), nullable=False),
    sa.Column('span_end', sa.Integer(), nullable=False),
    sa.Column('text', sa.Text(), nullable=False),
    sa.Column('claim_type', sa.String(length=20), nullable=False),
    sa.Column('metric', sa.String(length=40), nullable=True),
    sa.Column('value', sa.Numeric(precision=28, scale=6), nullable=True),
    sa.Column('value_high', sa.Numeric(precision=28, scale=6), nullable=True),
    sa.Column('unit', sa.String(length=20), nullable=True),
    sa.Column('target_fiscal_year', sa.Integer(), nullable=True),
    sa.Column('target_fiscal_period', sa.String(length=2), nullable=True),
    sa.Column('comparison', sa.String(length=10), nullable=True),
    sa.Column('hedging_score', sa.Float(), nullable=False),
    sa.Column('speaker_role', sa.String(length=20), nullable=True),
    sa.Column('linked_claim_id', sa.Integer(), nullable=True),
    sa.Column('extractor_version', sa.String(length=40), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.CheckConstraint("claim_type IN ('reported_result', 'comparison', 'guidance', 'qualitative')", name='claim_type'),
    sa.CheckConstraint('(filing_id IS NULL) <> (segment_id IS NULL)', name='one_source'),
    sa.ForeignKeyConstraint(['company_id'], ['company.id'], ),
    sa.ForeignKeyConstraint(['filing_id'], ['filing.id'], ),
    sa.ForeignKeyConstraint(['linked_claim_id'], ['claim.id'], ),
    sa.ForeignKeyConstraint(['segment_id'], ['segment.id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('filing_id', 'segment_id', 'span_start', 'span_end', 'extractor_version', name='uq_claim_span', postgresql_nulls_not_distinct=True)
    )
    op.create_index(op.f('ix_claim_company_id'), 'claim', ['company_id'], unique=False)
    op.create_table('evidence',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('report_id', sa.UUID(), nullable=False),
    sa.Column('claim_id', sa.Integer(), nullable=False),
    sa.Column('agent', sa.String(length=20), nullable=False),
    sa.Column('stance', sa.String(length=15), nullable=False),
    sa.Column('summary', sa.Text(), nullable=False),
    sa.Column('citations', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.Column('payload', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.Column('model', sa.String(length=40), nullable=True),
    sa.Column('input_tokens', sa.Integer(), nullable=False),
    sa.Column('output_tokens', sa.Integer(), nullable=False),
    sa.Column('cache_read_tokens', sa.Integer(), nullable=False),
    sa.Column('cost_usd', sa.Float(), nullable=False),
    sa.Column('latency_ms', sa.Integer(), nullable=False),
    sa.Column('trace_id', sa.UUID(), nullable=False),
    sa.CheckConstraint("agent IN ('filing_rag', 'xbrl_table', 'forecast', 'market_reaction', 'critic')", name='agent'),
    sa.CheckConstraint("stance IN ('supports', 'contradicts', 'insufficient')", name='stance'),
    sa.ForeignKeyConstraint(['claim_id'], ['claim.id'], ),
    sa.ForeignKeyConstraint(['report_id'], ['report.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_evidence_claim_id'), 'evidence', ['claim_id'], unique=False)
    op.create_table('verdict',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('report_id', sa.UUID(), nullable=False),
    sa.Column('claim_id', sa.Integer(), nullable=False),
    sa.Column('label', sa.String(length=15), nullable=False),
    sa.Column('confidence', sa.Float(), nullable=False),
    sa.Column('rationale', sa.Text(), nullable=False),
    sa.Column('citations', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.Column('severity', sa.Float(), nullable=False),
    sa.Column('refused', sa.Boolean(), nullable=False),
    sa.Column('prompt_version', sa.String(length=40), nullable=False),
    sa.Column('model', sa.String(length=40), nullable=True),
    sa.Column('cost_usd', sa.Float(), nullable=False),
    sa.Column('trace_id', sa.UUID(), nullable=False),
    sa.CheckConstraint("label IN ('supported', 'contradicted', 'unverifiable')", name='label'),
    sa.CheckConstraint('confidence BETWEEN 0 AND 1', name='confidence'),
    sa.ForeignKeyConstraint(['claim_id'], ['claim.id'], ),
    sa.ForeignKeyConstraint(['report_id'], ['report.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('report_id', 'claim_id', name='uq_verdict')
    )
    

def downgrade() -> None:
    op.drop_table('verdict')
    op.drop_index(op.f('ix_evidence_claim_id'), table_name='evidence')
    op.drop_table('evidence')
    op.drop_index(op.f('ix_claim_company_id'), table_name='claim')
    op.drop_table('claim')
    op.drop_index('ix_report_created', table_name='report')
    op.drop_table('report')
    op.drop_index(op.f('ix_span_trace_id'), table_name='span')
    op.drop_table('span')
    op.drop_table('eval_run')
