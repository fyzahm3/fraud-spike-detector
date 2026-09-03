"""Persistence-hardening tests for ReviewQueue (connection policy + indexes).

Covers the failure mode that would show up in a live demo: app.py builds a
fresh ReviewQueue per HTTP request, so a reviewer resolving an item can overlap
with a batch run enqueuing new ones. Without WAL and a busy timeout that raises
sqlite3.OperationalError("database is locked").
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

import pytest

from src.explain.queue import ReviewQueue
from src.explain.risk_brief import ContributingFactor, RiskBrief


def make_brief(i: int, score: float | None = None) -> RiskBrief:
    return RiskBrief(
        entity_id=f"card_{i}",
        flagged_type="transaction",
        model_score=score if score is not None else 0.5 + (i % 50) / 100.0,
        top_factors=[ContributingFactor(feature="g_card_cnt_24h", value=float(i),
                                        direction="increases_risk")],
        confidence="high",
        estimated_fp_cost=100.0 + i,
        recommended_action="hold_for_review",
        summary_text=f"Synthetic brief {i} for persistence testing.",
    )


@pytest.fixture()
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "review_queue.db"


# --------------------------------------------------------------------------
# Connection policy
# --------------------------------------------------------------------------

def test_connection_uses_wal_and_enforces_foreign_keys(db_path: Path):
    """A fresh connection must report WAL journaling and FK enforcement on."""
    ReviewQueue(db_path=db_path)

    with sqlite3.connect(str(db_path)) as raw:
        # journal_mode is a persistent property of the file itself.
        assert raw.execute("PRAGMA journal_mode;").fetchone()[0].lower() == "wal"

    # foreign_keys is per-connection, so assert on a connection the queue made.
    queue = ReviewQueue(db_path=db_path)
    conn = queue._get_conn()
    try:
        assert conn.execute("PRAGMA foreign_keys;").fetchone()[0] == 1
        assert conn.execute("PRAGMA busy_timeout;").fetchone()[0] >= 5000
        assert conn.execute("PRAGMA journal_mode;").fetchone()[0].lower() == "wal"
    finally:
        conn.close()


def test_query_indexes_exist(db_path: Path):
    """Both query patterns (pending list, audit lookup) must be indexed."""
    ReviewQueue(db_path=db_path)
    with sqlite3.connect(str(db_path)) as raw:
        names = {r[0] for r in raw.execute(
            "SELECT name FROM sqlite_master WHERE type = 'index';").fetchall()}
    assert "idx_review_queue_pending" in names
    assert "idx_audit_log_queue_id" in names


def test_pending_query_uses_the_index(db_path: Path):
    """Guard against the index silently not matching the query's shape."""
    queue = ReviewQueue(db_path=db_path)
    for i in range(5):
        queue.enqueue(make_brief(i))
    conn = queue._get_conn()
    try:
        plan = " ".join(str(r[-1]) for r in conn.execute(
            "EXPLAIN QUERY PLAN SELECT id FROM review_queue WHERE status = 'pending' "
            "ORDER BY model_score DESC, created_at ASC, id ASC;").fetchall())
    finally:
        conn.close()
    assert "idx_review_queue_pending" in plan, f"index not used; plan was: {plan}"


def test_schema_survives_database_file_deletion(db_path: Path):
    """Once-per-path init must not cache away the ability to self-heal."""
    queue = ReviewQueue(db_path=db_path)
    queue.enqueue(make_brief(1))

    for suffix in ("", "-wal", "-shm"):
        p = Path(str(db_path) + suffix)
        if p.exists():
            p.unlink()

    rebuilt = ReviewQueue(db_path=db_path)          # same path, file now gone
    assert rebuilt.list_pending() == []
    assert rebuilt.enqueue(make_brief(2)) > 0        # schema was recreated


# --------------------------------------------------------------------------
# Concurrency
# --------------------------------------------------------------------------

def test_concurrent_enqueue_and_resolve_never_locks(db_path: Path):
    """Many threads writing the same database must not raise 'database is locked'."""
    seed_queue = ReviewQueue(db_path=db_path)
    seeded_ids = [seed_queue.enqueue(make_brief(i)) for i in range(20)]

    n_writers, per_writer = 8, 10
    errors: list[BaseException] = []
    enqueued: list[int] = []
    lock = threading.Lock()
    start = threading.Barrier(n_writers)

    def writer(w: int) -> None:
        try:
            start.wait(timeout=10)
            queue = ReviewQueue(db_path=db_path)   # each thread its own, as app.py does
            for k in range(per_writer):
                qid = queue.enqueue(make_brief(w * 100 + k))
                with lock:
                    enqueued.append(qid)
        except BaseException as exc:                # noqa: BLE001 - recorded and re-raised below
            with lock:
                errors.append(exc)

    def resolver(w: int) -> None:
        try:
            start.wait(timeout=10)
            queue = ReviewQueue(db_path=db_path)
            for qid in seeded_ids[w::4]:
                queue.resolve(qid, reviewer_action="resolved_true_positive",
                              note=f"resolved by worker {w}")
        except BaseException as exc:                # noqa: BLE001
            with lock:
                errors.append(exc)

    threads = ([threading.Thread(target=writer, args=(w,)) for w in range(4)]
               + [threading.Thread(target=resolver, args=(w,)) for w in range(4)])
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)

    assert not any(t.is_alive() for t in threads), "a worker thread deadlocked"
    assert not errors, f"concurrent access raised: {errors[0]!r}"

    # Every write landed exactly once, and ids are unique.
    assert len(enqueued) == 4 * per_writer
    assert len(set(enqueued)) == len(enqueued)

    check = ReviewQueue(db_path=db_path)
    with sqlite3.connect(str(db_path)) as raw:
        total = raw.execute("SELECT COUNT(*) FROM review_queue;").fetchone()[0]
        resolved = raw.execute(
            "SELECT COUNT(*) FROM review_queue WHERE status != 'pending';").fetchone()[0]
        audit_rows = raw.execute("SELECT COUNT(*) FROM audit_log;").fetchone()[0]

    assert total == 20 + 4 * per_writer
    assert resolved == len(seeded_ids)
    assert audit_rows == len(seeded_ids)             # one audit row per resolve
    assert len(check.list_pending()) == total - len(seeded_ids)
    for qid in seeded_ids:
        assert len(check.get_audit_log(qid)) == 1


def test_concurrent_resolves_of_same_item_all_append(db_path: Path):
    """Append-only: repeated resolves of one item must not lose audit history."""
    queue = ReviewQueue(db_path=db_path)
    qid = queue.enqueue(make_brief(1))

    errors: list[BaseException] = []
    lock = threading.Lock()
    start = threading.Barrier(6)

    def resolve_once(w: int) -> None:
        try:
            start.wait(timeout=10)
            ReviewQueue(db_path=db_path).resolve(
                qid, reviewer_action="escalated", note=f"note {w}")
        except BaseException as exc:                # noqa: BLE001
            with lock:
                errors.append(exc)

    threads = [threading.Thread(target=resolve_once, args=(w,)) for w in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)

    assert not errors, f"concurrent resolve raised: {errors[0]!r}"
    log = queue.get_audit_log(qid)
    assert len(log) == 6, "audit rows were overwritten instead of appended"
    assert {entry["note"] for entry in log} == {f"note {w}" for w in range(6)}


# --------------------------------------------------------------------------
# Pagination (defaults must be a no-op)
# --------------------------------------------------------------------------

def test_list_pending_defaults_unchanged(db_path: Path):
    """Regression guard: no-arg list_pending still returns every pending row."""
    queue = ReviewQueue(db_path=db_path)
    for i in range(12):
        queue.enqueue(make_brief(i, score=0.90 - i * 0.01))

    rows = queue.list_pending()
    assert len(rows) == 12
    scores = [r["model_score"] for r in rows]
    assert scores == sorted(scores, reverse=True)
    assert rows[0]["model_score"] == pytest.approx(0.90)
    # Shape is unchanged: parsed factors in, raw json column out.
    assert "top_factors" in rows[0] and "top_factors_json" not in rows[0]
    assert rows[0]["top_factors"][0]["feature"] == "g_card_cnt_24h"


def test_pagination_slices_and_preserves_order(db_path: Path):
    queue = ReviewQueue(db_path=db_path)
    for i in range(12):
        queue.enqueue(make_brief(i, score=0.90 - i * 0.01))

    everything = queue.list_pending()

    assert queue.list_pending(limit=5) == everything[:5]
    assert queue.list_pending(limit=5, offset=5) == everything[5:10]
    assert queue.list_pending(limit=5, offset=10) == everything[10:]
    assert queue.list_pending(offset=4) == everything[4:]     # offset without limit
    assert queue.list_pending(limit=0) == []
    assert queue.list_pending(limit=100) == everything        # over-large limit
    assert queue.list_pending(offset=999) == []

    # Walking the whole queue in pages reconstructs it exactly, no gaps or dupes.
    paged: list[dict] = []
    page_size, offset = 5, 0
    while True:
        page = queue.list_pending(limit=page_size, offset=offset)
        if not page:
            break
        paged.extend(page)
        offset += page_size
    assert paged == everything


def test_pagination_rejects_negative_arguments(db_path: Path):
    queue = ReviewQueue(db_path=db_path)
    with pytest.raises(ValueError):
        queue.list_pending(limit=-1)
    with pytest.raises(ValueError):
        queue.list_pending(offset=-1)


def test_ties_are_ordered_deterministically(db_path: Path):
    """Equal scores in the same second must still page stably (id tiebreak)."""
    queue = ReviewQueue(db_path=db_path)
    ids = [queue.enqueue(make_brief(i, score=0.75)) for i in range(8)]
    assert [r["id"] for r in queue.list_pending()] == ids
    assert [r["id"] for r in queue.list_pending(limit=3, offset=3)] == ids[3:6]


# --------------------------------------------------------------------------
# Contracts that must survive the change
# --------------------------------------------------------------------------

def test_resolve_unknown_id_still_raises_keyerror(db_path: Path):
    """FK enforcement must not turn the KeyError contract into IntegrityError."""
    queue = ReviewQueue(db_path=db_path)
    with pytest.raises(KeyError):
        queue.resolve(999_999, reviewer_action="resolved_true_positive", note="nope")

    with sqlite3.connect(str(db_path)) as raw:
        assert raw.execute("SELECT COUNT(*) FROM audit_log;").fetchone()[0] == 0


def test_foreign_key_blocks_orphan_audit_row(db_path: Path):
    """The FK is actually enforced, not merely declared."""
    queue = ReviewQueue(db_path=db_path)
    conn = queue._get_conn()
    try:
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO audit_log (queue_id, reviewer_action, note) "
                "VALUES (?, ?, ?);", (424242, "escalated", "orphan"))
            conn.commit()
    finally:
        conn.close()


def test_public_api_is_still_exactly_four_methods():
    """Mirrors the defense-only introspection test — hardening added no surface."""
    public = {m for m in dir(ReviewQueue)
              if not m.startswith("_") and callable(getattr(ReviewQueue, m))}
    assert public == {"enqueue", "list_pending", "resolve", "get_audit_log"}


def test_reader_not_blocked_by_open_write_transaction(db_path: Path):
    """The concrete failure WAL fixes: dashboard reads during a batch write.

    In the default rollback-journal mode a writer whose transaction has spilled
    to disk holds an exclusive lock, and a concurrent reader gets
    "database is locked". Under WAL the reader sees the last committed snapshot
    and proceeds. Verified against journal_mode=DELETE, where this fails.
    """
    queue = ReviewQueue(db_path=db_path)
    for i in range(5):
        queue.enqueue(make_brief(i))

    reader_result: dict[str, object] = {}
    writing = threading.Event()
    may_commit = threading.Event()

    def hold_write_transaction() -> None:
        # Must be opened inside this thread: sqlite3 connections are thread-bound.
        writer = sqlite3.connect(str(db_path), timeout=5.0)
        writer.execute("PRAGMA journal_mode = WAL;")
        writer.execute("BEGIN IMMEDIATE;")
        # Enough rows to spill the page cache to disk and take the hard lock.
        writer.executemany(
            "INSERT INTO review_queue (entity_id, flagged_type, model_score,"
            " confidence, estimated_fp_cost, recommended_action, summary_text,"
            " top_factors_json, status) VALUES (?, 'transaction', 0.5, 'high',"
            " 1.0, 'hold_for_review', ?, '[]', 'pending');",
            [(f"bulk_{n}", "x" * 500) for n in range(8000)])
        writing.set()
        may_commit.wait(timeout=10)
        writer.commit()
        writer.close()

    t = threading.Thread(target=hold_write_transaction)
    t.start()
    try:
        assert writing.wait(timeout=15), "writer never opened its transaction"
        try:
            # Must return the pre-transaction snapshot, not raise.
            reader_result["rows"] = ReviewQueue(db_path=db_path).list_pending()
        except sqlite3.OperationalError as exc:      # pragma: no cover - regression path
            reader_result["error"] = exc
    finally:
        may_commit.set()
        t.join(timeout=15)

    assert "error" not in reader_result, (
        f"reader blocked by concurrent write: {reader_result.get('error')!r}")
    assert len(reader_result["rows"]) == 5      # uncommitted bulk rows not visible
