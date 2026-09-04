/* ==========================================================================
   Fraud-Spike Review Queue — console behaviour
   --------------------------------------------------------------------------
   Every value from the API reaches the page through textContent. Nothing is
   ever assigned to innerHTML: summary_text is LLM-generated and the factor
   fields carry model feature names, so none of it is trusted markup.
   ========================================================================== */

(function () {
    "use strict";

    var DECISIONS = [
        ["resolved_false_positive", "Dismiss as false positive", "btn--dismiss"],
        ["escalated", "Escalate", "btn--escalate"],
        ["resolved_true_positive", "Confirm fraud", "btn--primary"]
    ];

    var DECISION_LABELS = {
        resolved_true_positive: "Confirmed fraud",
        resolved_false_positive: "False positive",
        escalated: "Escalated"
    };

    var MAX_NOTE_LENGTH = 2000;

    var state = { pending: [], audit: [], view: "pending" };

    /* --- tiny DOM helpers ------------------------------------------------ */

    function el(tag, options, children) {
        var node = document.createElement(tag);
        options = options || {};
        if (options.className) { node.className = options.className; }
        if (options.id) { node.id = options.id; }
        if (options.text !== undefined && options.text !== null) {
            node.textContent = String(options.text);
        }
        if (options.attrs) {
            Object.keys(options.attrs).forEach(function (key) {
                node.setAttribute(key, options.attrs[key]);
            });
        }
        (children || []).forEach(function (child) { node.appendChild(child); });
        return node;
    }

    function fill(container, nodes) {
        while (container.firstChild) { container.removeChild(container.firstChild); }
        (nodes || []).forEach(function (node) { container.appendChild(node); });
    }

    function text(value) { return document.createTextNode(String(value)); }

    function byId(id) { return document.getElementById(id); }

    function money(value) {
        return Number(value).toLocaleString("en-US", {
            style: "currency", currency: "USD", maximumFractionDigits: 0
        });
    }

    function csrfToken() {
        var meta = document.querySelector('meta[name="csrf-token"]');
        return meta ? meta.getAttribute("content") : "";
    }

    /* --- status panels --------------------------------------------------- */

    function loadingPanel(message) {
        return el("div", { className: "panel panel--loading" }, [
            el("p", { text: message })
        ]);
    }

    function emptyPanel(heading, message) {
        return el("div", { className: "panel" }, [
            el("h3", { text: heading }),
            el("p", { text: message })
        ]);
    }

    function errorPanel(heading, message, detail, retry) {
        var nodes = [el("h3", { text: heading }), el("p", { text: message })];
        if (detail) {
            nodes.push(el("p", { className: "panel__detail", text: detail }));
        }
        var button = el("button", {
            className: "btn", text: "Try again", attrs: { type: "button" }
        });
        button.addEventListener("click", retry);
        nodes.push(button);
        return el("div", {
            className: "panel panel--error", attrs: { role: "alert" }
        }, nodes);
    }

    /* --- pending briefs -------------------------------------------------- */

    function confidenceTag(confidence) {
        var known = { high: 1, medium: 1, low: 1 };
        var suffix = known[confidence] ? confidence : "low";
        return el("span", {
            className: "tag tag--confidence-" + suffix,
            text: confidence + " confidence"
        });
    }

    function typeTag(flaggedType) {
        var isSpike = flaggedType === "spike";
        return el("span", {
            className: "tag " + (isSpike ? "tag--spike" : "tag--transaction"),
            text: isSpike ? "Spike event" : "Transaction"
        });
    }

    function factorRow(factor) {
        var value = typeof factor.value === "number"
            ? factor.value.toFixed(2)
            : factor.value;
        var increases = factor.direction === "increases_risk";
        return el("tr", {}, [
            el("td", { className: "factors__feature", text: factor.feature }),
            el("td", { className: "factors__value", text: value }),
            el("td", {
                className: "factors__direction " +
                    (increases ? "factors__direction--up" : "factors__direction--down"),
                text: increases ? "Increases risk" : "Reduces risk"
            })
        ]);
    }

    function factorsTable(factors) {
        var head = el("thead", {}, [
            el("tr", {}, [
                el("th", { text: "Model feature", attrs: { scope: "col" } }),
                el("th", { text: "Value", attrs: { scope: "col" } }),
                el("th", { text: "Direction", attrs: { scope: "col" } })
            ])
        ]);
        var body = el("tbody", {}, (factors || []).map(factorRow));
        var table = el("table", { className: "factors" }, [
            el("caption", { text: "Top contributing factors" }), head, body
        ]);
        return el("div", { className: "factors-wrap" }, [table]);
    }

    function actionBar(item) {
        var recommendation = el("p", { className: "brief__recommendation" });
        recommendation.appendChild(text("Model recommendation "));
        recommendation.appendChild(
            el("span", { className: "mono", text: item.recommended_action })
        );
        recommendation.appendChild(text(" · estimated cost if dismissed in error "));
        recommendation.appendChild(
            el("span", { className: "mono num", text: money(item.estimated_fp_cost) })
        );

        var noteId = "note-" + item.id;
        var input = el("input", {
            id: noteId,
            attrs: {
                type: "text",
                maxlength: String(MAX_NOTE_LENGTH),
                placeholder: "Recorded with the decision",
                autocomplete: "off"
            }
        });
        var noteField = el("div", { className: "note-field" }, [
            el("label", { text: "Reviewer note", attrs: { for: noteId } }),
            input
        ]);

        var buttons = DECISIONS.map(function (decision) {
            var button = el("button", {
                className: "btn " + decision[2],
                text: decision[1],
                attrs: { type: "button" }
            });
            button.addEventListener("click", function () {
                resolveItem(item.id, decision[0], input.value);
            });
            return button;
        });

        return el("footer", { className: "brief__actions" }, [
            recommendation,
            noteField,
            el("div", { className: "decisions" }, buttons)
        ]);
    }

    function briefBlock(item) {
        var head = el("div", { className: "brief__head" }, [
            el("div", { className: "brief__identity" }, [
                typeTag(item.flagged_type),
                el("span", { className: "brief__entity mono", text: "Entity " + item.entity_id }),
                confidenceTag(item.confidence)
            ]),
            el("div", { className: "brief__scoreblock" }, [
                el("span", { className: "brief__score num", text: item.model_score.toFixed(4) }),
                el("span", { className: "brief__score-label", text: "Risk score" })
            ])
        ]);

        var summary = el("p", { className: "brief__summary" }, [
            el("span", { className: "brief__summary-label", text: "Risk brief" })
        ]);
        summary.appendChild(text(item.summary_text));

        return el("article", {
            className: "brief",
            id: "item-" + item.id,
            attrs: { "aria-label": "Brief " + item.id }
        }, [head, summary, factorsTable(item.top_factors), actionBar(item)]);
    }

    /* --- audit trail ----------------------------------------------------- */

    function auditRow(entry) {
        var decision = DECISION_LABELS[entry.reviewer_action] || entry.reviewer_action;
        var note = entry.note
            ? el("td", { className: "audit__note", text: entry.note })
            : el("td", { className: "audit__note audit__note--empty", text: "No note recorded" });

        return el("tr", {}, [
            el("td", { className: "audit__id", text: "#" + entry.queue_id }),
            el("td", {}, [
                el("span", { className: "mono", text: entry.entity_id }),
                text(" "),
                typeTag(entry.flagged_type)
            ]),
            el("td", { className: "audit__score num", text: entry.model_score.toFixed(4) }),
            el("td", {}, [
                el("span", {
                    className: "decision decision--" + entry.reviewer_action,
                    text: decision
                })
            ]),
            note,
            el("td", { className: "audit__time", text: entry.timestamp })
        ]);
    }

    function auditTable(entries) {
        var columns = ["Item", "Entity", "Score", "Decision", "Reviewer note", "Recorded at"];
        var head = el("thead", {}, [
            el("tr", {}, columns.map(function (label) {
                return el("th", { text: label, attrs: { scope: "col" } });
            }))
        ]);
        var body = el("tbody", {}, entries.map(auditRow));
        return el("div", { className: "audit-wrap" }, [
            el("table", { className: "audit" }, [head, body])
        ]);
    }

    /* --- rendering ------------------------------------------------------- */

    function renderMetrics() {
        var items = state.pending;
        byId("metric-pending").textContent = items.length;
        byId("count-pending").textContent = items.length;
        byId("metric-resolved").textContent = state.audit.length;
        byId("count-audit").textContent = state.audit.length;

        if (items.length === 0) {
            byId("metric-score").textContent = "—";
            byId("metric-cost").textContent = "—";
            byId("metric-score-note").textContent = " ";
            byId("metric-pending-note").textContent = "Queue clear";
            return;
        }

        var total = items.reduce(function (sum, item) { return sum + item.model_score; }, 0);
        var cost = items.reduce(function (sum, item) { return sum + item.estimated_fp_cost; }, 0);
        var lowest = items.reduce(function (min, item) {
            return Math.min(min, item.model_score);
        }, Infinity);
        var spikes = items.filter(function (item) {
            return item.flagged_type === "spike";
        }).length;

        byId("metric-score").textContent = (total / items.length).toFixed(4);
        byId("metric-cost").textContent = money(cost);
        byId("metric-score-note").textContent = "Lowest in queue " + lowest.toFixed(4);
        byId("metric-pending-note").textContent =
            spikes + (spikes === 1 ? " spike event" : " spike events") + ", " +
            (items.length - spikes) + " single";
    }

    function renderPending() {
        var container = byId("pending-container");
        container.setAttribute("aria-busy", "false");
        if (state.pending.length === 0) {
            fill(container, [emptyPanel(
                "No briefs awaiting review",
                "Every flagged transaction and spike event in this queue has a recorded decision. " +
                "The audit trail holds the full history."
            )]);
            return;
        }
        fill(container, [el("div", { className: "briefs" }, state.pending.map(briefBlock))]);
    }

    function renderAudit() {
        var container = byId("audit-container");
        if (state.audit.length === 0) {
            fill(container, [emptyPanel(
                "No decisions recorded yet",
                "Resolve a brief in the pending view and it will appear here, with the reviewer " +
                "note and the time it was written."
            )]);
            return;
        }
        fill(container, [auditTable(state.audit)]);
    }

    /* --- data ------------------------------------------------------------ */

    function request(url) {
        return fetch(url, { headers: { Accept: "application/json" } })
            .then(function (res) {
                if (!res.ok) {
                    throw new Error("The server returned HTTP " + res.status + " for " + url + ".");
                }
                return res.json();
            });
    }

    function load() {
        byId("pending-container").setAttribute("aria-busy", "true");
        fill(byId("pending-container"), [loadingPanel("Loading review queue…")]);
        fill(byId("audit-container"), [loadingPanel("Loading audit trail…")]);

        return Promise.all([request("/api/pending"), request("/api/audit")])
            .then(function (results) {
                state.pending = results[0];
                state.audit = results[1];
                renderMetrics();
                renderPending();
                renderAudit();
            })
            .catch(function (error) {
                state.pending = [];
                state.audit = [];
                var panel = function () {
                    return errorPanel(
                        "Could not load the review queue",
                        "The console is running, but the queue database did not answer. Pending " +
                        "briefs and recorded decisions are unchanged — nothing was lost.",
                        error.message,
                        load
                    );
                };
                byId("pending-container").setAttribute("aria-busy", "false");
                fill(byId("pending-container"), [panel()]);
                fill(byId("audit-container"), [panel()]);
                ["metric-pending", "metric-score", "metric-cost", "metric-resolved"]
                    .forEach(function (id) { byId(id).textContent = "—"; });
                byId("metric-pending-note").textContent = "Unavailable";
                byId("metric-score-note").textContent = " ";
            });
    }

    function resolveItem(queueId, action, note) {
        var card = byId("item-" + queueId);
        if (card) {
            card.classList.add("brief--resolving");
            Array.prototype.forEach.call(card.querySelectorAll("button, input"),
                function (control) { control.disabled = true; });
        }

        fetch("/api/resolve/" + queueId, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "X-CSRF-Token": csrfToken()
            },
            body: JSON.stringify({ action: action, note: note || "" })
        }).then(function (res) {
            if (res.ok) { return load(); }
            return res.json().catch(function () { return {}; }).then(function (body) {
                throw new Error(body.error || "The server returned HTTP " + res.status + ".");
            });
        }).catch(function (error) {
            if (card) {
                card.classList.remove("brief--resolving");
                Array.prototype.forEach.call(card.querySelectorAll("button, input"),
                    function (control) { control.disabled = false; });
                var existing = card.querySelector(".panel--error");
                if (existing) { existing.remove(); }
                card.appendChild(errorPanel(
                    "Decision not recorded",
                    "Nothing was written to the audit log. The brief is still pending.",
                    error.message,
                    function () { resolveItem(queueId, action, note); }
                ));
            }
        });
    }

    /* --- view switching --------------------------------------------------- */

    function showView(name) {
        state.view = name;
        [["pending", "tab-pending", "view-pending"], ["audit", "tab-audit", "view-audit"]]
            .forEach(function (entry) {
                var active = entry[0] === name;
                byId(entry[1]).setAttribute("aria-selected", active ? "true" : "false");
                byId(entry[2]).hidden = !active;
            });
    }

    Array.prototype.forEach.call(document.querySelectorAll(".view-tab"), function (tab) {
        tab.addEventListener("click", function () { showView(tab.dataset.view); });
    });

    showView("pending");
    load();
})();
