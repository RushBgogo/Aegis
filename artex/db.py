"""任务作用域 + 数据库原子状态转换；SQLite 默认，PostgreSQL 可选。"""
from contextlib import contextmanager
from copy import deepcopy
import os
from pathlib import Path
import time
from uuid import uuid4

from sqlalchemy import (
    Column, Float, ForeignKey, Integer, JSON, MetaData, String, Table,
    Text, UniqueConstraint, create_engine, event, select, update, delete,
)
from sqlalchemy.exc import IntegrityError
from sqlalchemy.pool import StaticPool

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATABASE_URL = f"sqlite:///{(PROJECT_ROOT / 'data' / 'study.db').as_posix()}"

metadata = MetaData()
tasks = Table("tasks", metadata,
    Column("id", String(64), primary_key=True), Column("description", Text, nullable=False),
    Column("goal", Text, nullable=False), Column("scope", JSON, nullable=False),
    Column("status", String(24), nullable=False), Column("version", Integer, nullable=False),
    Column("created_at", Float, nullable=False), Column("deadline", Float))
nodes = Table("nodes", metadata,
    Column("id", String(64), primary_key=True),
    Column("task_id", String(64), ForeignKey("tasks.id"), nullable=False),
    Column("kind", String(24), nullable=False), Column("text", Text, nullable=False),
    Column("state", String(24), nullable=False), Column("priority", Integer, nullable=False),
    Column("dedup_key", String(256), nullable=False), Column("payload", JSON, nullable=False),
    Column("version", Integer, nullable=False), Column("created_at", Float, nullable=False),
    UniqueConstraint("task_id", "dedup_key"))
edges = Table("edges", metadata,
    Column("id", String(64), primary_key=True),
    Column("task_id", String(64), ForeignKey("tasks.id"), nullable=False),
    Column("source_id", String(64), ForeignKey("nodes.id"), nullable=False),
    Column("target_id", String(64), ForeignKey("nodes.id"), nullable=False),
    Column("relation", String(32), nullable=False),
    UniqueConstraint("task_id", "source_id", "target_id", "relation"))
events = Table("events", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("task_id", String(64), ForeignKey("tasks.id"), nullable=False),
    Column("kind", String(64), nullable=False), Column("data", JSON, nullable=False),
    Column("call_id", String(128)), Column("created_at", Float, nullable=False))
records = Table("records", metadata,
    Column("task_id", String(64), primary_key=True),
    Column("namespace", String(64), primary_key=True), Column("key", String(256), primary_key=True),
    Column("value", JSON, nullable=False), Column("version", Integer, nullable=False))

TASK_TRANSITIONS = {
    "created": {"running", "paused", "cancelled", "deleting"},
    "running": {"paused", "done", "failed", "cancelled", "deleting"},
    "paused": {"running", "cancelled", "deleting"},
    "done": {"running", "deleting"}, "failed": {"running", "deleting"}, "cancelled": {"deleting"},
    "deleting": {"deleted"}, "deleted": set(),
}


class Repository:
    def __init__(self, url=None):
        url = url or os.getenv("ARTEX_DATABASE_URL") or DEFAULT_DATABASE_URL
        if url.startswith("sqlite") and ":memory:" not in url and ":///" in url:
            prefix, database_value = url.split(":///", 1)
            database = Path(database_value)
            if not database.is_absolute():
                database = PROJECT_ROOT / database
            database.parent.mkdir(parents=True, exist_ok=True)
            url = f"{prefix}:///{database.as_posix()}"
        options = {}
        if url.startswith("sqlite"):
            options["connect_args"] = {"check_same_thread": False, "timeout": 15}
            if ":memory:" in url:
                options["poolclass"] = StaticPool
        self.engine = create_engine(url, **options)
        if url.startswith("sqlite"):
            @event.listens_for(self.engine, "connect")
            def enable_foreign_keys(connection, _):
                connection.execute("PRAGMA foreign_keys=ON")

    def setup(self):
        """仅用于隔离测试库；正式学习数据库使用 alembic upgrade head。"""
        metadata.create_all(self.engine)

    def close(self):
        self.engine.dispose()

    @contextmanager
    def write(self):
        with self.engine.connect() as conn:
            try:
                if conn.dialect.name == "sqlite":
                    conn.exec_driver_sql("BEGIN IMMEDIATE")
                else:
                    conn.begin()
                yield conn
                conn.commit()
            except BaseException:
                conn.rollback()
                raise

    def _task(self, conn, task_id, write=False):
        query = select(tasks).where(tasks.c.id == task_id)
        if write:
            query = query.with_for_update()
        row = conn.execute(query).mappings().first()
        if row is None:
            raise KeyError("task not found")
        if write and row["status"] in {"deleting", "deleted"}:
            raise ValueError("task write barrier")
        return dict(row)

    def create_task(self, description, goal, scope=None, task_id=None, deadline=None):
        value = dict(id=task_id or uuid4().hex, description=description, goal=goal,
                     scope=list(scope or []), status="created", version=1,
                     created_at=time.time(), deadline=deadline)
        with self.write() as conn:
            conn.execute(tasks.insert().values(**value))
        return value

    def get_task(self, task_id):
        with self.engine.connect() as conn:
            return self._task(conn, task_id)

    def list_tasks(self):
        with self.engine.connect() as conn:
            return [dict(row) for row in conn.execute(select(tasks)).mappings()]

    def transition_task(self, task_id, expected, target):
        if target not in TASK_TRANSITIONS.get(expected, set()):
            raise ValueError(f"invalid task transition: {expected}->{target}")
        with self.write() as conn:
            result = conn.execute(update(tasks).where(tasks.c.id == task_id,
                tasks.c.status == expected).values(status=target, version=tasks.c.version + 1))
            return result.rowcount == 1

    def set_deadline(self, task_id, deadline):
        with self.write() as conn:
            self._task(conn, task_id, write=True)
            conn.execute(update(tasks).where(tasks.c.id == task_id).values(deadline=deadline))

    def create_node(self, task_id, kind, text, *, key=None, state=None, payload=None, priority=0):
        if kind not in {"goal", "intent", "fact", "finding", "hint", "digest"}:
            raise ValueError("invalid node kind")
        value = dict(id=uuid4().hex, task_id=task_id, kind=kind, text=text,
            dedup_key=key or uuid4().hex, state=state or ("open" if kind in {"goal", "intent"} else "confirmed"),
            priority=priority, payload=deepcopy(payload or {}), version=1, created_at=time.time())
        with self.write() as conn:
            self._task(conn, task_id, write=True)
            old = conn.execute(select(nodes).where(nodes.c.task_id == task_id,
                nodes.c.dedup_key == value["dedup_key"])).mappings().first()
            if old:
                if old["kind"] != kind:
                    raise ValueError("dedup key belongs to another node kind")
                return dict(old)
            conn.execute(nodes.insert().values(**value))
        return value

    def get_node(self, task_id, node_id):
        with self.engine.connect() as conn:
            row = conn.execute(select(nodes).where(nodes.c.id == node_id,
                nodes.c.task_id == task_id)).mappings().first()
            if row is None:
                raise KeyError("node not found in task")
            return dict(row)

    def list_nodes(self, task_id, kind=None, state=None):
        query = select(nodes).where(nodes.c.task_id == task_id)
        if kind is not None:
            query = query.where(nodes.c.kind == kind)
        if state is not None:
            query = query.where(nodes.c.state == state)
        with self.engine.connect() as conn:
            return [dict(row) for row in conn.execute(query.order_by(
                nodes.c.priority.desc(), nodes.c.created_at)).mappings()]

    def claim_intent(self, task_id, node_id, owner):
        with self.write() as conn:
            task = self._task(conn, task_id, write=True)
            if task["status"] != "running":
                return False
            row = conn.execute(select(nodes).where(nodes.c.task_id == task_id,
                nodes.c.id == node_id, nodes.c.kind == "intent")).mappings().first()
            if not row:
                return False
            payload = {**row["payload"], "owner": owner}
            result = conn.execute(update(nodes).where(nodes.c.id == node_id,
                nodes.c.state == "open").values(state="running", payload=payload,
                    version=nodes.c.version + 1))
            return result.rowcount == 1

    def transition_node(self, task_id, node_id, expected, target):
        with self.write() as conn:
            self._task(conn, task_id, write=True)
            result = conn.execute(update(nodes).where(nodes.c.id == node_id,
                nodes.c.task_id == task_id, nodes.c.state == expected).values(
                    state=target, version=nodes.c.version + 1))
            return result.rowcount == 1

    def update_node(self, task_id, node_id, text, payload, expected_version):
        with self.write() as conn:
            self._task(conn, task_id, write=True)
            result = conn.execute(update(nodes).where(nodes.c.id == node_id,
                nodes.c.task_id == task_id, nodes.c.version == expected_version).values(
                    text=text, payload=deepcopy(payload), version=nodes.c.version + 1))
            return result.rowcount == 1

    def add_edge(self, task_id, source_id, target_id, relation):
        if relation not in {"spawns", "derived_from", "yields", "proves", "covers"}:
            raise ValueError("invalid edge relation")
        with self.write() as conn:
            self._task(conn, task_id, write=True)
            for node_id in (source_id, target_id):
                if conn.execute(select(nodes.c.id).where(nodes.c.id == node_id,
                    nodes.c.task_id == task_id)).first() is None:
                    raise ValueError("edge crosses task boundary")
            condition = (edges.c.task_id == task_id, edges.c.source_id == source_id,
                         edges.c.target_id == target_id, edges.c.relation == relation)
            if not conn.execute(select(edges.c.id).where(*condition)).first():
                conn.execute(edges.insert().values(id=uuid4().hex, task_id=task_id,
                    source_id=source_id, target_id=target_id, relation=relation))

    def list_edges(self, task_id):
        with self.engine.connect() as conn:
            return [dict(row) for row in conn.execute(select(edges).where(
                edges.c.task_id == task_id)).mappings()]

    def append_event(self, task_id, kind, data, *, call_id=None):
        value = dict(task_id=task_id, kind=kind, data=deepcopy(data),
                     call_id=call_id, created_at=time.time())
        with self.write() as conn:
            self._task(conn, task_id, write=True)
            result = conn.execute(events.insert().values(**value))
            value["id"] = result.inserted_primary_key[0]
        return value

    def list_events(self, task_id, since=0, limit=100):
        with self.engine.connect() as conn:
            return [dict(row) for row in conn.execute(select(events).where(
                events.c.task_id == task_id, events.c.id > since).order_by(
                    events.c.id).limit(max(1, min(limit, 1000)))).mappings()]

    def get_record(self, namespace, key, task_id=""):
        with self.engine.connect() as conn:
            row = conn.execute(select(records.c.value, records.c.version).where(
                records.c.task_id == task_id, records.c.namespace == namespace,
                records.c.key == key)).mappings().first()
            return dict(row) if row else None

    def put_record(self, namespace, key, value, task_id="", expected_version=0):
        try:
            with self.write() as conn:
                if task_id:
                    self._task(conn, task_id, write=True)
                if expected_version == 0:
                    conn.execute(records.insert().values(task_id=task_id,
                        namespace=namespace, key=key, value=deepcopy(value), version=1))
                    return True
                result = conn.execute(update(records).where(records.c.task_id == task_id,
                    records.c.namespace == namespace, records.c.key == key,
                    records.c.version == expected_version).values(value=deepcopy(value),
                        version=records.c.version + 1))
                return result.rowcount == 1
        except IntegrityError:
            return False

    def list_records(self, namespace, task_id=""):
        with self.engine.connect() as conn:
            return [dict(row) for row in conn.execute(select(records.c.key,
                records.c.value, records.c.version).where(records.c.task_id == task_id,
                    records.c.namespace == namespace).order_by(records.c.key)).mappings()]

    def delete_record(self, namespace, key, task_id="", expected_version=1):
        with self.write() as conn:
            if task_id:
                self._task(conn, task_id, write=True)
            result = conn.execute(delete(records).where(records.c.task_id == task_id,
                records.c.namespace == namespace, records.c.key == key,
                records.c.version == expected_version))
            return result.rowcount == 1
