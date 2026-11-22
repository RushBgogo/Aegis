"""固定的第一版业务 schema；后续新表必须写新 revision。"""
from alembic import op
import sqlalchemy as sa


revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    op.create_table('records',
    sa.Column('task_id', sa.String(length=64), nullable=False),
    sa.Column('namespace', sa.String(length=64), nullable=False),
    sa.Column('key', sa.String(length=256), nullable=False),
    sa.Column('value', sa.JSON(), nullable=False),
    sa.Column('version', sa.Integer(), nullable=False),
    sa.PrimaryKeyConstraint('task_id', 'namespace', 'key')
    )
    op.create_table('tasks',
    sa.Column('id', sa.String(length=64), nullable=False),
    sa.Column('description', sa.Text(), nullable=False),
    sa.Column('goal', sa.Text(), nullable=False),
    sa.Column('scope', sa.JSON(), nullable=False),
    sa.Column('status', sa.String(length=24), nullable=False),
    sa.Column('version', sa.Integer(), nullable=False),
    sa.Column('created_at', sa.Float(), nullable=False),
    sa.Column('deadline', sa.Float(), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('events',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('task_id', sa.String(length=64), nullable=False),
    sa.Column('kind', sa.String(length=64), nullable=False),
    sa.Column('data', sa.JSON(), nullable=False),
    sa.Column('call_id', sa.String(length=128), nullable=True),
    sa.Column('created_at', sa.Float(), nullable=False),
    sa.ForeignKeyConstraint(['task_id'], ['tasks.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('nodes',
    sa.Column('id', sa.String(length=64), nullable=False),
    sa.Column('task_id', sa.String(length=64), nullable=False),
    sa.Column('kind', sa.String(length=24), nullable=False),
    sa.Column('text', sa.Text(), nullable=False),
    sa.Column('state', sa.String(length=24), nullable=False),
    sa.Column('priority', sa.Integer(), nullable=False),
    sa.Column('dedup_key', sa.String(length=256), nullable=False),
    sa.Column('payload', sa.JSON(), nullable=False),
    sa.Column('version', sa.Integer(), nullable=False),
    sa.Column('created_at', sa.Float(), nullable=False),
    sa.ForeignKeyConstraint(['task_id'], ['tasks.id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('task_id', 'dedup_key')
    )
    op.create_table('edges',
    sa.Column('id', sa.String(length=64), nullable=False),
    sa.Column('task_id', sa.String(length=64), nullable=False),
    sa.Column('source_id', sa.String(length=64), nullable=False),
    sa.Column('target_id', sa.String(length=64), nullable=False),
    sa.Column('relation', sa.String(length=32), nullable=False),
    sa.ForeignKeyConstraint(['source_id'], ['nodes.id'], ),
    sa.ForeignKeyConstraint(['target_id'], ['nodes.id'], ),
    sa.ForeignKeyConstraint(['task_id'], ['tasks.id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('task_id', 'source_id', 'target_id', 'relation')
    )


def downgrade():
    op.drop_table('edges')
    op.drop_table('nodes')
    op.drop_table('events')
    op.drop_table('tasks')
    op.drop_table('records')
