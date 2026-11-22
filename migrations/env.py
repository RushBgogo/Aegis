import os

from alembic import context

from artex.db import Repository, metadata

url = os.getenv("ARTEX_DATABASE_URL") or context.config.get_main_option("sqlalchemy.url")
repository = Repository(url)
try:
    with repository.engine.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=metadata,
            render_as_batch=connection.dialect.name == "sqlite",
        )
        with context.begin_transaction():
            context.run_migrations()
finally:
    repository.close()
