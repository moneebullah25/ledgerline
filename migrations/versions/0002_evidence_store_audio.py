"""evidence store and audio: pgvector chunks, call audio, segments

Revision ID: 0002
Revises: 0001
"""
from alembic import op
import sqlalchemy as sa
import pgvector.sqlalchemy
from sqlalchemy.dialects import postgresql

revision = "0002"
down_revision = '0001'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.create_table('call_audio',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('company_id', sa.Integer(), nullable=True),
    sa.Column('kind', sa.String(length=20), nullable=False),
    sa.Column('event_date', sa.Date(), nullable=True),
    sa.Column('fiscal_year', sa.Integer(), nullable=True),
    sa.Column('fiscal_period', sa.String(length=2), nullable=True),
    sa.Column('storage_path', sa.Text(), nullable=False),
    sa.Column('duration_s', sa.Float(), nullable=True),
    sa.Column('speech_s', sa.Float(), nullable=True),
    sa.Column('reference_transcript_path', sa.Text(), nullable=True),
    sa.Column('source_url', sa.Text(), nullable=False),
    sa.Column('fetched_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('sha256', sa.String(length=64), nullable=False),
    sa.Column('license_note', sa.Text(), nullable=False),
    sa.ForeignKeyConstraint(['company_id'], ['company.id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('sha256', name='uq_call_audio_sha')
    )
    op.create_table('chunk',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('filing_id', sa.Integer(), nullable=False),
    sa.Column('chunk_index', sa.Integer(), nullable=False),
    sa.Column('section_path', sa.Text(), nullable=False),
    sa.Column('char_start', sa.Integer(), nullable=False),
    sa.Column('char_end', sa.Integer(), nullable=False),
    sa.Column('text', sa.Text(), nullable=False),
    sa.Column('token_count', sa.Integer(), nullable=False),
    sa.Column('embedding', pgvector.sqlalchemy.vector.VECTOR(dim=384), nullable=True),
    sa.Column('tsv', postgresql.TSVECTOR(), sa.Computed("to_tsvector('english', text)", persisted=True), nullable=False),
    sa.ForeignKeyConstraint(['filing_id'], ['filing.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('filing_id', 'chunk_index', name='uq_chunk')
    )
    op.create_index('ix_chunk_embedding', 'chunk', ['embedding'], unique=False, postgresql_using='hnsw', postgresql_ops={'embedding': 'vector_cosine_ops'})
    op.create_index('ix_chunk_tsv', 'chunk', ['tsv'], unique=False, postgresql_using='gin')
    op.create_table('segment',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('call_audio_id', sa.Integer(), nullable=False),
    sa.Column('idx', sa.Integer(), nullable=False),
    sa.Column('t_start', sa.Float(), nullable=False),
    sa.Column('t_end', sa.Float(), nullable=False),
    sa.Column('speaker_label', sa.String(length=20), nullable=False),
    sa.Column('role', sa.String(length=20), nullable=False),
    sa.Column('section', sa.String(length=20), nullable=False),
    sa.Column('text', sa.Text(), nullable=False),
    sa.Column('words', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.CheckConstraint("role IN ('operator','executive','analyst','unknown')", name='role'),
    sa.CheckConstraint("section IN ('prepared_remarks','qa')", name='section'),
    sa.ForeignKeyConstraint(['call_audio_id'], ['call_audio.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('call_audio_id', 'idx', name='uq_segment')
    )

def downgrade() -> None:
    op.drop_table('segment')
    op.drop_index('ix_chunk_tsv', table_name='chunk', postgresql_using='gin')
    op.drop_index('ix_chunk_embedding', table_name='chunk', postgresql_using='hnsw', postgresql_ops={'embedding': 'vector_cosine_ops'})
    op.drop_table('chunk')
    op.drop_table('call_audio')
