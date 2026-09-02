"""SQLite-backed review queue and append-only audit trail (Phase 4).

Defense-only proof point: Stores flagged risk briefs for human review.
resolve() never deletes or overwrites rows — it appends to audit_log and updates
the status column only.
"""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import sqlite3
from typing import Literal

from src.explain.risk_brief import RiskBrief


class ReviewQueue:
    """Backed by SQLite (results/review_queue.db).

    Public API is strictly limited to four methods to enforce defense-only side-effect safety.
    """

    def __init__(self, db_path: str | Path = "results/review_queue.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._get_conn() as conn:
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
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
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
            conn.commit()

    def enqueue(self, brief: RiskBrief) -> int:
        """Enqueue a new RiskBrief for human review. Returns queue_id."""
        factors_json = json.dumps([asdict(f) for f in brief.top_factors])
        with self._get_conn() as conn:
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO review_queue (
                    entity_id, flagged_type, model_score, confidence,
                    estimated_fp_cost, recommended_action, summary_text,
                    top_factors_json, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending');
            """, (
                brief.entity_id,
                brief.flagged_type,
                brief.model_score,
                brief.confidence,
                brief.estimated_fp_cost,
                brief.recommended_action,
                brief.summary_text,
                factors_json,
            ))
            conn.commit()
            return int(cur.lastrowid)

    def list_pending(self) -> list[dict]:
        """List all pending items in the review queue."""
        with self._get_conn() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT id, entity_id, flagged_type, model_score, confidence,
                       estimated_fp_cost, recommended_action, summary_text,
                       top_factors_json, status, created_at
                FROM review_queue
                WHERE status = 'pending'
                ORDER BY model_score DESC, created_at ASC;
            """)
            rows = cur.fetchall()
            result = []
            for r in rows:
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

        with self._get_conn() as conn:
            cur = conn.cursor()
            # Verify queue item exists
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
            conn.commit()

    def get_audit_log(self, queue_id: int) -> list[dict]:
        """Fetch full audit history for a given queue item."""
        with self._get_conn() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT id, queue_id, reviewer_action, note, timestamp
                FROM audit_log
                WHERE queue_id = ?
                ORDER BY timestamp ASC, id ASC;
            """, (queue_id,))
            return [dict(r) for r in cur.fetchall()]
