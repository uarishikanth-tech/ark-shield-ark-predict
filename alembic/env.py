from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

# Make sure `app` is importable when alembic is run from the project root.
from app.config import settings
from app.database import Base

# Import every model module so Base.metadata is fully populated before
# autogenerate compares it against the database.
import app.models  # noqa: F401

config = context.config

# Override the ini file's connection string with our own sync DSN, so
# there's one source of truth (the .env file) rather than two places
# to keep in sync.
config.set_main_option("sqlalchemy.url", settings.alembic_database_url)

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
