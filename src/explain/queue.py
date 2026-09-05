"""SQLite-backed review queue and append-only audit trail (Phase 4).

Defense-only proof point: Stores flagged risk briefs for human review.
resolve() never deletes or overwrites rows — it appends to audit_log and updates
the status column only.

Connection policy (see _get_conn): every connection is opened in WAL mode with
a busy timeout and foreign-key enforcement on, because app.py builds a fresh
ReviewQueue per HTTP request and a reviewer resolving an item can overlap with
run_pipeline.py enqueuing new ones.
"""

from __future__ import annotations

from contextlib import closing
from dataclasses import asdict
import json
from pathlib import Path
import sqlite3
import threading
from typing import Literal

from src.explain.risk_brief import RiskBrief

# Seconds a blocked writer waits for a competing lock before raising
# "database is locked". Comfortably longer than any query this schema issues.
BUSY_TIMEOUT_SECONDS = 5.0

# Schema creation is idempotent but not free, and app.py constructs a
# ReviewQueue on every request. Track which database files have already been
# initialised in this process so the DDL runs once per path, not once per call.
_INITIALIZED_PATHS: set[str] = set()
_INIT_LOCK = threading.Lock()


class ReviewQueue:
    """Backed by SQLite (results/review_queue.db).

    Public API is strictly limited to four methods to enforce defense-only side-effect safety.
    """

    def __init__(self, db_path: str | Path = "results/review_queue.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _get_conn(self) -> sqlite3.Connection:
        """Open a connection configured for concurrent use.

        - journal_mode=WAL: the default rollback journal takes an exclusive lock
          for the whole write, so a reviewer loading the dashboard while a batch
          run is enqueuing briefs gets "database is locked". WAL puts writes in a
          separate log, letting readers continue against the last committed
          snapshot while one writer proceeds. It is a persistent property of the
          database file, but setting it per connection is cheap and idempotent
          and means a freshly created file is never left in rollback mode.
        - busy_timeout: writers serialise under WAL, so a second writer must
          still wait its turn; without a timeout it fails immediately instead.
        - foreign_keys=ON: SQLite ignores FK declarations unless asked per
          connection, so audit_log.queue_id -> review_queue(id) is only really
          enforced here. This makes an orphaned audit row impossible at the
          storage layer, not just by convention.
        """
        conn = sqlite3.connect(str(self.db_path), timeout=BUSY_TIMEOUT_SECONDS)
        conn.row_factory = sqlite3.Row
        conn.execute(f"PRAGMA busy_timeout = {int(BUSY_TIMEOUT_SECONDS * 1000)};")
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def _ensure_schema(self) -> None:
        """Create tables and indexes once per database path per process.

        The `exists()` check keeps the old self-healing behaviour: if the file
        is deleted underneath a long-running process, the next queue re-creates
        the schema rather than failing with "no such table".
        """
        key = str(self.db_path.resolve())
        with _INIT_LOCK:
            if key in _INITIALIZED_PATHS and self.db_path.exists():
                return
            self._init_db()
            _INITIALIZED_PATHS.add(key)

    def _init_db(self) -> None:
        with closing(self._get_conn()) as conn, conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS review_queue (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    entity_id TEXT NOT NULL,
                    flagged_type TEXT NOT NULL,
                    model_score REAL NOT NULL,
                    confidence TEXT NOT NULL,
                    estimated_fp_cost REAL NOT NULL,
                    recommended_action TEXT NOT NULL,
                    summary_text TEXT NOT NULL,
                    top_factors_json TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    gateway_score REAL
                );
            """)

            # Additive migration for databases created before the gateway model
            # existed — the committed demo snapshot among them. Nullable, so
            # every existing row keeps meaning exactly what it meant: no
            # gateway score was computed for it.
            existing = {row[1] for row in conn.execute("PRAGMA table_info(review_queue);")}
            if "gateway_score" not in existing:
                conn.execute("ALTER TABLE review_queue ADD COLUMN gateway_score REAL;")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    queue_id INTEGER NOT NULL,
                    reviewer_action TEXT NOT NULL,
                    note TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(queue_id) REFERENCES review_queue(id)
                );
            """)
            # Mirrors list_pending()'s filter and sort exactly, so the pending
            # view is an index scan rather than a full table sort.
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_review_queue_pending
                ON review_queue (status, model_score DESC, created_at ASC, id ASC);
            """)
            # Mirrors get_audit_log()'s filter and sort.
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_audit_log_queue_id
                ON audit_log (queue_id, timestamp ASC, id ASC);
            """)

    def enqueue(self, brief: RiskBrief) -> int:
        """Enqueue a new RiskBrief for human review. Returns queue_id."""
        factors_json = json.dumps([asdict(f) for f in brief.top_factors])
        with closing(self._get_conn()) as conn, conn:
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO review_queue (
                    entity_id, flagged_type, model_score, confidence,
                    estimated_fp_cost, recommended_action, summary_text,
                    top_factors_json, gateway_score, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending');
            """, (
                brief.entity_id,
                brief.flagged_type,
                brief.model_score,
                brief.confidence,
                brief.estimated_fp_cost,
                brief.recommended_action,
                brief.summary_text,
                factors_json,
                getattr(brief, "gateway_score", None),
            ))
            return int(cur.lastrowid)

    def list_pending(
        self,
        limit: int | None = None,
        offset: int = 0,
        status: str = "pending",
    ) -> list[dict]:
        """List review-queue items with the given status, highest risk first.

        Defaults (limit=None, offset=0, status="pending") return every pending
        row — identical to the pre-pagination behaviour, so existing callers are
        unaffected. `id ASC` is a final tiebreak: created_at has one-second
        granularity, so without it rows enqueued in the same second have no
        stable order and a paged read could skip or repeat one.

        `status` is a signature extension rather than a second method, because
        the public surface of this class is fixed at four methods and asserted
        by tests/test_explain.py::test_no_blocking_side_effects. The dashboard's
        audit-trail view needs resolved rows; it gets them here. The name stays
        `list_pending` for the same reason — renaming it would break the API
        assertion just as surely as adding to it.

        Passing a status the queue never writes returns an empty list rather
        than raising: an unknown status is a query that matches nothing, not a
        programming error.
        """
        if limit is not None and limit < 0:
            raise ValueError(f"limit must be non-negative, got {limit}")
        if offset < 0:
            raise ValueError(f"offset must be non-negative, got {offset}")
        if not isinstance(status, str) or not status:
            raise ValueError(f"status must be a non-empty string, got {status!r}")

        # Parameterised, so the (status, model_score DESC, created_at ASC, id ASC)
        # index still serves every status the same way it served the literal.
        sql = """
            SELECT id, entity_id, flagged_type, model_score, confidence,
                   estimated_fp_cost, recommended_action, summary_text,
                   top_factors_json, gateway_score, status, created_at
            FROM review_queue
            WHERE status = ?
            ORDER BY model_score DESC, created_at ASC, id ASC
        """
        params: list[object] = [status]
        if limit is not None:
            sql += " LIMIT ? OFFSET ?"
            params += [int(limit), int(offset)]
        elif offset:
            # SQLite requires a LIMIT before OFFSET; -1 means "no limit".
            sql += " LIMIT -1 OFFSET ?"
            params += [int(offset)]

        with closing(self._get_conn()) as conn:
            cur = conn.cursor()
            cur.execute(sql + ";", params)
            result = []
            for r in cur.fetchall():
                item = dict(r)
                item["top_factors"] = json.loads(item.pop("top_factors_json"))
                result.append(item)
            return result

    def resolve(
        self,
        queue_id: int,
        reviewer_action: Literal["resolved_true_positive", "resolved_false_positive", "escalated"],
        note: str = "",
    ) -> None:
        """Resolve a queued item.

        Append-only audit trail: never deletes or overwrites original row content;
        inserts an audit_log record and updates status column.
        """
        valid_actions = {"resolved_true_positive", "resolved_false_positive", "escalated"}
        if reviewer_action not in valid_actions:
            raise ValueError(f"Invalid reviewer_action: '{reviewer_action}'. Must be one of {valid_actions}")

        with closing(self._get_conn()) as conn, conn:
            cur = conn.cursor()
            # Verify queue item exists. Foreign keys would also reject an
            # orphaned audit row, but as IntegrityError — this check keeps the
            # caller-facing KeyError contract.
            cur.execute("SELECT id FROM review_queue WHERE id = ?;", (queue_id,))
            if not cur.fetchone():
                raise KeyError(f"Queue ID {queue_id} not found in review_queue.")

            # Update status
            cur.execute("UPDATE review_queue SET status = ? WHERE id = ?;", (reviewer_action, queue_id))
            # Insert audit log entry
            cur.execute("""
                INSERT INTO audit_log (queue_id, reviewer_action, note)
                VALUES (?, ?, ?);
            """, (queue_id, reviewer_action, str(note)))

    def get_audit_log(self, queue_id: int) -> list[dict]:
        """Fetch full audit history for a given queue item."""
        with closing(self._get_conn()) as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT id, queue_id, reviewer_action, note, timestamp
                FROM audit_log
                WHERE queue_id = ?
                ORDER BY timestamp ASC, id ASC;
            """, (queue_id,))
            return [dict(r) for r in cur.fetchall()]
