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
        ["resolved_false_positive", "Dismiss as false positive", "btn-quiet"],
        ["escalated", "Escalate", "btn-quiet"],
        ["resolved_true_positive", "Confirm fraud", "btn-primary"]
    ];

    var DECISION_LABELS = {
        resolved_true_positive: "Confirmed fraud",
        resolved_false_positive: "False positive",
        escalated: "Escalated"
    };

    var MAX_NOTE_LENGTH = 2000;

    /* An item ingested live from the payment rail. The model never scored it,
       because a webhook payload does not carry the feature space the model was
       trained on. Everything below keys off this value rather than off a
       missing score, so an unscored item can never fall through to a scored
       item's rendering by accident. */
    var UNSCORED_TYPE = "live_demo_unscored";

    function isScored(item) {
        return item.flagged_type !== UNSCORED_TYPE;
    }

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
        return el("div", { className: "state state-loading" }, [
            el("span", { className: "spinner", attrs: { "aria-hidden": "true" } }),
            el("p", { text: message })
        ]);
    }

    function emptyPanel(heading, message) {
        return el("div", { className: "state" }, [
            el("h3", { text: heading }),
            el("p", { text: message })
        ]);
    }

    function errorPanel(heading, message, detail, retry) {
        var nodes = [el("h3", { text: heading }), el("p", { text: message })];
        if (detail) {
            nodes.push(el("p", { className: "state-detail", text: detail }));
        }
        var button = el("button", {
            className: "btn", text: "Try again", attrs: { type: "button" }
        });
        button.addEventListener("click", retry);
        nodes.push(button);
        return el("div", {
            className: "state state-error", attrs: { role: "alert" }
        }, nodes);
    }

    /* --- pending briefs -------------------------------------------------- */

    function confidenceTag(confidence) {
        var known = { high: 1, medium: 1, low: 1 };
        var suffix = known[confidence] ? confidence : "low";
        return el("span", { className: "tag", text: confidence + " confidence" });
    }

    function typeTag(flaggedType) {
        /* Three distinct labels, distinguished by wording and not only by
           colour: a viewer who cannot see colour must still be unable to
           mistake a live unscored item for a scored one. */
        if (flaggedType === UNSCORED_TYPE) {
            return el("span", {
                className: "tag tag-unscored",
                text: "Live ingestion · gateway score only"
            });
        }
        var isSpike = flaggedType === "spike";
        return el("span", {
            className: "tag " + (isSpike ? "tag-accent" : ""),
            text: isSpike ? "Spike event" : "Transaction"
        });
    }

    function factorRow(factor) {
        var value = typeof factor.value === "number"
            ? factor.value.toFixed(2)
            : factor.value;
        /* A live-ingested item's fields are observations, not model inputs,
           and must not read as risk evidence in either direction. */
        if (factor.direction === "gateway_model_input") {
            return el("tr", {}, [
                el("td", { className: "factors-feature", text: factor.feature }),
                el("td", { className: "factors-value", text: value }),
                el("td", { className: "factors-gateway", text: "Gateway model input" })
            ]);
        }
        if (factor.direction === "not_a_model_input") {
            return el("tr", {}, [
                el("td", { className: "factors-feature", text: factor.feature }),
                el("td", { className: "factors-value", text: value }),
                el("td", {
                    className: "factors-neutral",
                    text: "Not a model input"
                })
            ]);
        }
        var increases = factor.direction === "increases_risk";
        return el("tr", {}, [
            el("td", { className: "factors-feature", text: factor.feature }),
            el("td", { className: "factors-value", text: value }),
            el("td", {
                className: "" +
                    (increases ? "factors-up" : "factors-down"),
                text: increases ? "Increases risk" : "Reduces risk"
            })
        ]);
    }

    function factorsTable(factors, scored) {
        /* The column and caption wording changes too, not just the cells: for an
           unscored item these rows are payload fields, and calling them
           contributing factors would be the same false claim in another place. */
        var head = el("thead", {}, [
            el("tr", {}, [
                el("th", {
                    text: scored ? "Model feature" : "Payload field",
                    attrs: { scope: "col" }
                }),
                el("th", { text: "Value", attrs: { scope: "col" } }),
                el("th", {
                    text: scored ? "Direction" : "Role",
                    attrs: { scope: "col" }
                })
            ])
        ]);
        var body = el("tbody", {}, (factors || []).map(factorRow));
        var table = el("table", { className: "factors" }, [
            el("caption", {
                text: scored
                    ? "Top contributing factors"
                    : "Observed payload fields, and which the gateway model used"
            }),
            head,
            body
        ]);
        return el("div", { className: "factors-wrap" }, [table]);
    }

    function actionBar(item) {
        var recommendation = el("p", { className: "brief-rec" });
        if (isScored(item)) {
            recommendation.appendChild(text("Model recommendation "));
            recommendation.appendChild(
                el("span", { className: "mono", text: item.recommended_action })
            );
            recommendation.appendChild(text(" \u00b7 estimated cost if dismissed in error "));
            recommendation.appendChild(
                el("span", { className: "mono num", text: money(item.estimated_fp_cost) })
            );
        } else {
            /* No "model recommendation" line here. The model made no
               recommendation about this item, and a $0 cost estimate would be a
               number standing in for a judgement that was never made. */
            recommendation.appendChild(text(
                "The model made no recommendation for this item and estimated no cost. " +
                "It is queued so a human can see the ingested payment; any decision " +
                "recorded is entirely the reviewer's own."
            ));
        }

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

        return el("footer", { className: "brief-ft" }, [
            recommendation,
            noteField,
            el("div", { className: "decisions" }, buttons)
        ]);
    }

    function briefBlock(item) {
        var scored = isScored(item);

        /* The confidence tag is suppressed for an unscored item: there is no
           scored judgement to be more or less confident about, so the word
           should not appear anywhere near it. */
        var identity = [
            typeTag(item.flagged_type),
            el("span", { className: "brief-entity mono", text: "Entity " + item.entity_id })
        ];
        if (scored) { identity.push(confidenceTag(item.confidence)); }

        /* Where a scored item shows four decimal places, this shows words. The
           server already sends model_score as null for these, so there is no
           number here to accidentally format. */
        var scoreblock;
        if (scored) {
            scoreblock = el("div", { className: "brief-scoreblock" }, [
                el("span", { className: "brief-score num", text: formatScore(item.model_score) }),
                el("span", { className: "brief-score-label", text: "Risk score \u00b7 full model" })
            ]);
        } else if (item.gateway) {
            /* A real score from a real model — the gateway one — labelled so it
               can never be read as the full model's number. */
            scoreblock = el("div", { className: "brief-scoreblock" }, [
                el("span", {
                    className: "brief-score brief-score-gateway num",
                    text: formatScore(item.gateway.score)
                }),
                el("span", { className: "brief-score-label", text: "Gateway score" })
            ]);
        } else {
            scoreblock = el("div", { className: "brief-scoreblock" }, [
                el("span", { className: "brief-score-none", text: "Not scored" }),
                el("span", { className: "brief-score-label", text: "No model score exists" })
            ]);
        }

        var head = el("div", { className: "brief-hd" }, [
            el("div", { className: "brief-id" }, identity),
            scoreblock
        ]);

        var summary = el("p", { className: "brief-summary" }, [
            el("span", {
                className: "brief-summary-label",
                text: scored ? "Risk brief" : "Ingestion note"
            })
        ]);
        summary.appendChild(text(item.summary_text));

        var gatewayPanel = null;
        if (!scored && item.gateway) {
            var g = item.gateway;
            var rows = el("div", { className: "gwrows" }, [
                scoreRow("Gateway decision", g.score >= g.threshold
                    ? "Above threshold \u2014 send for review"
                    : "Below threshold \u2014 no action indicated"),
                scoreRow("Gateway threshold", formatScore(g.threshold)),
                scoreRow("Features available now", String(g.n_features)),
                scoreRow("Gateway AUC-PR", formatScore(g.auc_pr)),
                scoreRow("Full model AUC-PR", formatScore(g.full_model_auc_pr)
                    + " (on " + g.full_model_n_features + " features)")
            ]);
            gatewayPanel = el("div", { className: "gateway-panel" }, [
                el("p", { className: "gateway-title", text: "Scored at authorization, by the gateway model" }),
                el("p", {
                    className: "gateway-body",
                    text: "At the instant this payment was authorized it had no history yet " +
                          "\u2014 no device graph, no email-cluster signal, none of the entity " +
                          "relationships the full model draws most of its strength from. This " +
                          "score comes from a model trained on the same dataset restricted to " +
                          "the fields a webhook actually carries, and it is correspondingly " +
                          "weaker. The gap between the two figures below is the measured cost " +
                          "of the history that has not accumulated yet."
                }),
                rows
            ]);
        }

        return el("article", {
            className: "brief" + (scored ? "" : " brief-unscored"),
            id: "item-" + item.id,
            attrs: { "aria-label": "Brief " + item.id }
        }, [
            head,
            /* .brief-body carries the horizontal padding. Without it the
               summary and the factor table sit flush against the card border,
               which is what made these cards read as cramped. */
            el("div", { className: "brief-body" }, gatewayPanel
                ? [summary, gatewayPanel, factorsTable(item.top_factors, scored)]
                : [summary, factorsTable(item.top_factors, scored)]),
            actionBar(item)
        ]);
    }

    /* --- audit trail ----------------------------------------------------- */

    /* A resolved live item still has the gateway model's score, and hiding it
       behind "Not scored" was wrong — the reviewer saw a number when they made
       the call, so the record of that call has to show the same number, marked
       as the gateway model's. */
    function auditScoreCell(entry) {
        if (entry.scored !== false && entry.model_score !== null) {
            return el("td", { className: "audit-score num", text: formatScore(entry.model_score) });
        }
        if (entry.gateway && typeof entry.gateway.score === "number") {
            return el("td", { className: "audit-score num" }, [
                el("span", { text: formatScore(entry.gateway.score) }),
                el("span", { className: "audit-score-tag", text: "gateway" })
            ]);
        }
        return el("td", { className: "audit-score audit-score-none", text: "Not scored" });
    }

    function auditRow(entry) {
        var decision = DECISION_LABELS[entry.reviewer_action] || entry.reviewer_action;
        var note = entry.note
            ? el("td", { className: "audit-note", text: entry.note })
            : el("td", { className: "audit-note audit-note-empty", text: "No note recorded" });

        return el("tr", { attrs: { "data-queue-id": String(entry.queue_id) } }, [
            el("td", { className: "audit-id", text: "#" + entry.queue_id }),
            el("td", {}, [
                el("span", { className: "mono", text: entry.entity_id }),
                text(" "),
                typeTag(entry.flagged_type)
            ]),
            auditScoreCell(entry),
            el("td", {}, [
                el("span", {
                    className: "decision decision--" + entry.reviewer_action,
                    text: decision
                })
            ]),
            note,
            el("td", { className: "audit-time", text: entry.timestamp })
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

        /* Every aggregate is computed over scored items only. An unscored item
           has no score to average and no cost estimate to add, and letting one
           into these sums would put a fabricated number on the page by the back
           door — the same failure as printing a fake score on the card. */
        var scoredItems = items.filter(isScored);
        var unscored = items.length - scoredItems.length;

        var cost = scoredItems.reduce(function (sum, item) {
            return sum + item.estimated_fp_cost;
        }, 0);
        var spikes = scoredItems.filter(function (item) {
            return item.flagged_type === "spike";
        }).length;

        byId("metric-cost").textContent = money(cost);

        if (scoredItems.length === 0) {
            byId("metric-score").textContent = "\u2014";
            byId("metric-score-note").textContent = "No scored items in queue";
        } else {
            var total = scoredItems.reduce(function (sum, item) {
                return sum + item.model_score;
            }, 0);
            var lowest = scoredItems.reduce(function (min, item) {
                return Math.min(min, item.model_score);
            }, Infinity);
            byId("metric-score").textContent = formatScore(total / scoredItems.length);
            byId("metric-score-note").textContent = "Lowest in queue " + formatScore(lowest) +
                (unscored ? " \u00b7 " + unscored + " unscored excluded" : "");
        }

        var composition =
            spikes + (spikes === 1 ? " spike event" : " spike events") + ", " +
            (scoredItems.length - spikes) + " single";
        if (unscored) {
            composition += ", " + unscored + " live unscored";
        }
        byId("metric-pending-note").textContent = composition;
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

    /* --- live ingestion (proof-of-concept) ------------------------------- */

    var CHECKOUT_SCRIPT = "https://checkout.razorpay.com/v1/checkout.js";
    var checkoutLoading = null;

    function liveStatus(message, isError) {
        var node = byId("live-trigger-status");
        if (!node) { return; }
        node.textContent = message;
        node.className = "live-trigger-status" +
            (isError ? " live-trigger-status-error" : "");
    }

    /* Loaded on demand rather than in the page head: this third-party script is
       only needed by whoever clicks the button, so an ordinary reviewer's
       session never fetches it. */
    function loadCheckout() {
        if (window.Razorpay) { return Promise.resolve(); }
        if (checkoutLoading) { return checkoutLoading; }
        checkoutLoading = new Promise(function (resolve, reject) {
            var script = document.createElement("script");
            script.src = CHECKOUT_SCRIPT;
            script.onload = function () { resolve(); };
            script.onerror = function () {
                checkoutLoading = null;
                reject(new Error("Could not load Razorpay's checkout script."));
            };
            document.head.appendChild(script);
        });
        return checkoutLoading;
    }

    function openCheckout(order) {
        return loadCheckout().then(function () {
            var checkout = new window.Razorpay({
                key: order.key_id,
                order_id: order.order_id,
                amount: order.amount,
                currency: order.currency,
                name: "Fraud-Spike Review Queue",
                description: "Test-mode live ingestion proof-of-concept",
                handler: function () {
                    /* Razorpay's webhook is the source of truth, not this
                       callback: the queue item is created server-side only
                       after an HMAC-verified delivery. Reloading here just
                       picks it up once it lands. */
                    liveStatus(
                        "Payment submitted. Waiting for the signed webhook \u2014 the item " +
                        "appears below once its signature verifies.", false
                    );
                    window.setTimeout(refreshIngested, 2500);
                },
                modal: {
                    ondismiss: function () {
                        liveStatus("Checkout closed. No payment was made.", false);
                    }
                }
            });
            checkout.open();
            liveStatus("Test-mode order " + order.order_id + " created. Complete it with a test card.", false);
        });
    }

    /* The trigger lives on /live but the queue it feeds is rendered on /demo,
       so the refresh has to find whichever container this page actually has. */
    function refreshIngested() {
        if (byId("live-items-container")) { return loadLiveItems(); }
        if (byId("pending-container")) { return load(); }
        return Promise.resolve();
    }

    function triggerLiveTransaction() {
        var button = byId("btn-live-trigger");
        if (button) { button.disabled = true; }
        liveStatus("Creating a test-mode order\u2026", false);

        fetch("/api/live/trigger", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "X-CSRF-Token": csrfToken()
            },
            body: "{}"
        }).then(function (res) {
            return res.json().catch(function () { return {}; }).then(function (body) {
                if (!res.ok) {
                    throw new Error(body.error || "The server returned HTTP " + res.status + ".");
                }
                return openCheckout(body);
            });
        }).catch(function (error) {
            liveStatus(error.message, true);
        }).then(function () {
            if (button) { button.disabled = false; }
        });
    }

    /* --- data ------------------------------------------------------------ */

    /* --- cold start ------------------------------------------------------

       The instance sleeps after inactivity on a free tier, and the first
       request afterwards can take the better part of a minute. Two things must
       not happen while that is true: the page must not sit there looking
       broken, and it must not hang forever with no way out.

       The honest limit of this code: once the container is fully asleep, the
       server cannot serve this script either, so the browser shows its own
       blank tab until the first byte arrives. Nothing in the page can fix that
       — the external uptime ping is what keeps the instance warm. What this
       does cover is every request after the shell has loaded.
       --------------------------------------------------------------------- */

    var SLOW_REQUEST_MS = 2500;   // past this, say something rather than spin silently
    var REQUEST_TIMEOUT_MS = 75000;  // past this, fail visibly with a retry

    var pendingRequests = 0;
    var slowTimer = null;

    function wakingBanner() {
        var existing = byId("waking-banner");
        if (existing) { return existing; }
        var banner = el("div", {
            className: "waking", id: "waking-banner",
            attrs: { role: "status", hidden: "hidden" }
        }, [
            el("span", { className: "spinner", attrs: { "aria-hidden": "true" } }),
            el("div", {}, [
                el("p", { className: "waking-title", text: "Waking the server\u2026" }),
                el("p", {
                    className: "waking-note",
                    text: "This instance sleeps after a period of inactivity, and the first " +
                          "request wakes it. It usually takes about 30 seconds. Nothing is wrong."
                })
            ])
        ]);
        var main = byId("main");
        if (main) { main.insertBefore(banner, main.firstChild); }
        return banner;
    }

    function showWaking() {
        var banner = wakingBanner();
        if (banner) { banner.hidden = false; }
    }

    function hideWaking() {
        var banner = byId("waking-banner");
        if (banner) { banner.hidden = true; }
    }

    function requestStarted() {
        pendingRequests += 1;
        if (slowTimer === null) {
            slowTimer = window.setTimeout(showWaking, SLOW_REQUEST_MS);
        }
    }

    function requestFinished() {
        pendingRequests = Math.max(0, pendingRequests - 1);
        if (pendingRequests === 0) {
            if (slowTimer !== null) { window.clearTimeout(slowTimer); slowTimer = null; }
            hideWaking();
        }
    }

    function request(url) {
        requestStarted();

        /* AbortController is the difference between "slow" and "hung". Without
           it a stalled connection leaves the page spinning with no error and no
           retry, which is the failure being designed out here. */
        var controller = window.AbortController ? new window.AbortController() : null;
        var timeout = window.setTimeout(function () {
            if (controller) { controller.abort(); }
        }, REQUEST_TIMEOUT_MS);

        var options = { headers: { Accept: "application/json" } };
        if (controller) { options.signal = controller.signal; }

        return fetch(url, options)
            .then(function (res) {
                if (!res.ok) {
                    throw new Error("The server returned HTTP " + res.status + " for " + url + ".");
                }
                return res.json();
            })
            .catch(function (error) {
                if (error && error.name === "AbortError") {
                    throw new Error(
                        "The server did not respond within " +
                        Math.round(REQUEST_TIMEOUT_MS / 1000) + " seconds. It may still be " +
                        "waking up \u2014 try again."
                    );
                }
                throw error;
            })
            .then(function (value) {
                window.clearTimeout(timeout); requestFinished(); return value;
            }, function (error) {
                window.clearTimeout(timeout); requestFinished(); throw error;
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

    /* --- recording a decision ---------------------------------------------

       A resolution is the one destructive-looking thing a reviewer does here,
       and the first version of this gave almost no sign it had happened: the
       card left a thirty-item list and a count changed on a tab nobody was
       looking at. That reads as a dead button.

       So a recorded decision now says so three times over — the card is
       replaced in place by a confirmation naming the decision, a toast appears
       with a route to the consequence, and the audit trail flashes the row that
       was written. The append-only log is the strongest thing this system has;
       the interface should point at it rather than hide it.
       --------------------------------------------------------------------- */

    function toast(message, linkLabel, onLink) {
        var host = byId("toast-host");
        if (!host) {
            host = el("div", { id: "toast-host", className: "toast-host" });
            document.body.appendChild(host);
        }
        var node = el("div", { className: "toast", attrs: { role: "status" } }, [
            el("span", { className: "toast-text", text: message })
        ]);
        if (linkLabel && onLink) {
            var link = el("button", {
                className: "toast-link", text: linkLabel, attrs: { type: "button" }
            });
            link.addEventListener("click", function () {
                onLink();
                if (node.parentNode) { node.parentNode.removeChild(node); }
            });
            node.appendChild(link);
        }
        host.appendChild(node);
        window.setTimeout(function () {
            node.classList.add("is-out");
            window.setTimeout(function () {
                if (node.parentNode) { node.parentNode.removeChild(node); }
            }, 400);
        }, 6000);
    }

    function showAuditTrail(flashQueueId) {
        /* The audit trail lives on the review queue. Anywhere else — the live
           ingestion page, for one — the honest thing is to go there rather than
           silently do nothing. */
        if (!showView("audit")) {
            var target = "/demo#audit";
            if (flashQueueId !== undefined) { target += "-" + flashQueueId; }
            window.location.href = target;
            return;
        }
        var panel = byId("view-audit");
        if (panel) { panel.scrollIntoView({ block: "start" }); }
        if (flashQueueId === undefined) { return; }
        var row = document.querySelector('[data-queue-id="' + flashQueueId + '"]');
        if (row) {
            row.classList.add("is-flash");
            window.setTimeout(function () { row.classList.remove("is-flash"); }, 2400);
        }
    }

    function confirmationCard(queueId, action, note) {
        var label = DECISION_LABELS[action] || action;
        var nodes = [
            el("p", { className: "recorded-title", text: "Decision recorded: " + label }),
            el("p", {
                className: "recorded-note",
                text: note
                    ? "Reviewer note: " + note
                    : "No reviewer note was recorded with this decision."
            }),
            el("p", {
                className: "recorded-note",
                text: "Written to the append-only audit log as item #" + queueId +
                      ". It cannot be edited or deleted; a correction is a new row."
            })
        ];
        var view = el("button", {
            className: "btn btn-quiet", text: "View it in the audit trail",
            attrs: { type: "button" }
        });
        view.addEventListener("click", function () { showAuditTrail(queueId); });
        nodes.push(view);
        return el("div", { className: "recorded" }, nodes);
    }

    function resolveItem(queueId, action, note) {
        var card = byId("item-" + queueId);
        if (card) {
            card.classList.add("brief-resolving");
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
            if (res.ok) {
                /* Confirm in place first. The list is refreshed a moment later,
                   so the card is not yanked out from under the eye that is
                   still reading it. */
                if (card) {
                    card.classList.remove("brief-resolving");
                    card.classList.add("brief-recorded");
                    fill(card, [confirmationCard(queueId, action, note)]);
                }
                toast(
                    "Decision recorded: " + (DECISION_LABELS[action] || action),
                    "View audit trail",
                    function () { showAuditTrail(queueId); }
                );
                /* Hold before refreshing. Reloading immediately replaced the
                   confirmation within a few hundred milliseconds — the card
                   simply vanished again, which is the behaviour being fixed.
                   The pause is what makes the confirmation readable. */
                return new Promise(function (resolve) {
                    window.setTimeout(function () {
                        load().then(function () {
                            /* Counts move as a consequence of the decision, so
                               draw the eye to the one that changed. */
                            var tile = byId("metric-resolved");
                            if (tile) {
                                tile.classList.add("is-bumped");
                                window.setTimeout(function () {
                                    tile.classList.remove("is-bumped");
                                }, 1400);
                            }
                            resolve();
                        });
                    }, 2600);
                });
            }
            return res.json().catch(function () { return {}; }).then(function (body) {
                throw new Error(body.error || "The server returned HTTP " + res.status + ".");
            });
        }).catch(function (error) {
            if (card) {
                card.classList.remove("brief-resolving");
                Array.prototype.forEach.call(card.querySelectorAll("button, input"),
                    function (control) { control.disabled = false; });
                var existing = card.querySelector(".state-error");
                if (existing) { existing.remove(); }
                card.appendChild(errorPanel(
                    "Decision not recorded",
                    "Nothing was written to the audit log. The brief is still pending.",
                    error.message,
                    function () { resolveItem(queueId, action, note); }
                ));
            }
            toast("Decision NOT recorded \u2014 nothing was written to the audit log.");
        });
    }

    /* --- live-ingestion page: what has been ingested so far --------------- */

    function loadLiveItems() {
        var container = byId("live-items-container");
        if (!container) { return Promise.resolve(); }
        fill(container, [loadingPanel("Checking the queue\u2026")]);

        return request("/api/pending").then(function (items) {
            var live = items.filter(function (item) { return !isScored(item); });
            if (live.length === 0) {
                fill(container, [emptyPanel(
                    "No live items ingested yet",
                    "Trigger a test-mode transaction above and complete it with a test card. The " +
                    "item appears here once its webhook signature verifies."
                )]);
                return;
            }
            fill(container, [el("div", { className: "briefs" }, live.map(briefBlock))]);
        }).catch(function (error) {
            fill(container, [errorPanel(
                "Could not read the queue",
                "The page is running, but the queue database did not answer.",
                error.message,
                loadLiveItems
            )]);
        });
    }

    /* --- live scoring -----------------------------------------------------

       The model runs server-side; this only asks it and renders the answer.
       Every value is written with textContent, like everywhere else on the
       site, and the score is whatever the response carried — there is no
       client-side default to fall back on if the request fails.
       --------------------------------------------------------------------- */

    var STRATUM_LABELS = {
        top_fraud: "Highest-scoring fraud",
        borderline: "Borderline, near the threshold",
        false_positive: "False positive \u2014 flagged but legitimate",
        false_negative: "Missed fraud \u2014 not flagged",
        clear_legitimate: "Clearly legitimate"
    };

    var OUTCOME_LABELS = {
        true_positive: "Correct \u2014 fraud, and the model flagged it",
        true_negative: "Correct \u2014 legitimate, and the model did not flag it",
        false_positive: "Wrong \u2014 legitimate, but the model flagged it",
        false_negative: "Wrong \u2014 fraud, and the model missed it"
    };

    /* Scores are the most real numbers on this site, so they must not be
       rounded into looking fake. The top of the queue genuinely scores
       0.9999834, and four decimal places render that as "1.0000" — a value the
       model never produced and which reads, to anyone who knows the maths, as a
       hard-coded placeholder. Extremes therefore get more precision rather than
       a rounder-looking lie. */
    function formatScore(value) {
        if (typeof value !== "number" || !isFinite(value)) { return "\u2014"; }
        var rounded = Number(value.toFixed(4));
        if (rounded >= 1 && value < 1) { return value.toFixed(6); }
        if (rounded <= 0 && value > 0) { return value.toFixed(6); }
        return value.toFixed(4);
    }

    function money2(value) {
        return Number(value).toLocaleString("en-US", {
            style: "currency", currency: "USD", minimumFractionDigits: 2
        });
    }

    function scoreRow(label, value, className) {
        return el("div", { className: "verdict-row" }, [
            el("span", { className: "verdict-key", text: label }),
            el("span", { className: "verdict-val " + (className || ""), text: value })
        ]);
    }

    function renderScore(result) {
        var correct = result.outcome === "true_positive" || result.outcome === "true_negative";

        var head = el("div", { className: "verdict-hd" }, [
            el("span", {
                className: "tag " + (result.flagged ? "tag-accent" : ""),
                text: result.flagged ? "Flagged for review" : "Not flagged"
            }),
            el("span", {
                className: "tag " + (correct ? "tag-good" : "tag-warn"),
                text: correct ? "Model was right" : "Model was wrong"
            })
        ]);

        var score = el("div", { className: "verdict-scoreblock" }, [
            el("span", { className: "verdict-score num", text: formatScore(result.model_score) }),
            el("span", { className: "verdict-score-label", text: "Risk score, computed just now" })
        ]);

        var rows = el("div", { className: "verdict-rows" }, [
            scoreRow("Decision threshold", formatScore(result.threshold)),
            scoreRow("Transaction ID", String(result.transaction_id)),
            scoreRow("Amount", money2(result.amount)),
            scoreRow("Ground truth", result.actual_label === 1 ? "Fraud" : "Legitimate"),
            scoreRow("Outcome", OUTCOME_LABELS[result.outcome] || result.outcome,
                     correct ? "verdict-val-good" : "verdict-val-bad"),
            scoreRow("Features fed to the model", String(result.n_features)),
            scoreRow("Model variant", result.variant)
        ]);

        return el("div", { className: "verdict" + (correct ? "" : " verdict-wrong") },
                  [head, score, rows]);
    }

    function runScorer() {
        var select = byId("scorer-select");
        var button = byId("scorer-run");
        var container = byId("scorer-result");
        if (!select || !container) { return Promise.resolve(); }

        if (button) { button.disabled = true; }
        fill(container, [loadingPanel("Running the model\u2026")]);

        return request("/api/score/" + encodeURIComponent(select.value))
            .then(function (body) {
                fill(container, [renderScore(body.result)]);
            })
            .catch(function (error) {
                fill(container, [errorPanel(
                    "The model did not return a score",
                    "Nothing is shown in place of it \u2014 a number without the model " +
                    "behind it would read as a prediction.",
                    error.message,
                    runScorer
                )]);
            })
            .then(function () {
                if (button) { button.disabled = false; }
            });
    }

    function initScorer() {
        var select = byId("scorer-select");
        if (!select) { return; }

        request("/api/score/samples").then(function (body) {
            body.samples.forEach(function (sample) {
                var label = (STRATUM_LABELS[sample.stratum] || sample.stratum) +
                    " \u00b7 #" + sample.transaction_id + " \u00b7 " + money2(sample.amount);
                select.appendChild(el("option", { text: label, attrs: { value: sample.id } }));
            });
            var run = byId("scorer-run");
            if (run) { run.addEventListener("click", runScorer); }
        }).catch(function (error) {
            fill(byId("scorer-result"), [errorPanel(
                "Live scoring is unavailable",
                "The trained model could not be loaded on this instance. The measured " +
                "figures above are unaffected; they are read from committed files.",
                error.message,
                initScorer
            )]);
        });
    }

    /* --- view switching --------------------------------------------------- */

    /* Guarded on every element: the tabs exist on the review queue and nowhere
       else, and this used to be called from the live page too — where it threw
       on the first null and killed the click. */
    function hasTabs() {
        return !!(byId("tab-pending") && byId("tab-audit")
                  && byId("view-pending") && byId("view-audit"));
    }

    function showView(name) {
        if (!hasTabs()) { return false; }
        state.view = name;
        [["pending", "tab-pending", "view-pending"], ["audit", "tab-audit", "view-audit"]]
            .forEach(function (entry) {
                var active = entry[0] === name;
                byId(entry[1]).setAttribute("aria-selected", active ? "true" : "false");
                byId(entry[2]).hidden = !active;
            });
        return true;
    }

    /* --- contextual help --------------------------------------------------

       The affordance is a <details> element rendered by the server, so it opens
       and closes without JavaScript and the explanation is in the document from
       the first byte. This adds only the two behaviours the element does not
       give for free: one panel open at a time, and Escape to close. A note that
       covers the figure it explains is worse than no note.
       --------------------------------------------------------------------- */

    function closeHints(except) {
        Array.prototype.forEach.call(document.querySelectorAll("details.hint[open]"),
            function (node) { if (node !== except) { node.open = false; } });
    }

    /* --- explainer side panel -----------------------------------------
       On the console (app_shell.html), a '?' opens a real panel with room
       to read instead of the small inline popover. The content is the same
       server-rendered text the <details> element already carries — this
       only moves where it is displayed, so nothing here can say anything
       the page does not already contain, and there is still no fetch and
       no model behind it. Pages without the shared panel markup (the
       marketing site) keep the plain inline popover unchanged. */
    function openExplainer(hint) {
        var panel = byId("explainer-panel");
        var backdrop = byId("explainer-backdrop");
        if (!panel || !backdrop) { return false; }

        var titleNode = hint.querySelector(".hint-title");
        var bodyNode = hint.querySelector(".hint-panel");
        if (!titleNode || !bodyNode) { return false; }

        byId("explainer-title").textContent = titleNode.textContent;
        var paragraphs = bodyNode.querySelectorAll("p");
        fill(byId("explainer-body"), Array.prototype.map.call(paragraphs, function (p) {
            return el("p", { text: p.textContent });
        }));

        backdrop.hidden = false;
        panel.hidden = false;
        // Two frames so the browser has painted the hidden->visible change
        // before the transform transitions; skipping this makes the panel
        // appear instantly instead of sliding in on the first open.
        window.requestAnimationFrame(function () {
            window.requestAnimationFrame(function () { panel.classList.add("is-open"); });
        });
        panel.focus();
        return true;
    }

    function closeExplainer() {
        var panel = byId("explainer-panel");
        var backdrop = byId("explainer-backdrop");
        if (!panel || !panel.classList.contains("is-open")) { return; }
        panel.classList.remove("is-open");
        window.setTimeout(function () {
            panel.hidden = true;
            if (backdrop) { backdrop.hidden = true; }
        }, prefersReducedMotion() ? 0 : 260);
    }

    function initExplainer() {
        var panel = byId("explainer-panel");
        if (!panel) { return; }

        panel.setAttribute("tabindex", "-1");
        var closeBtn = byId("explainer-close");
        var backdrop = byId("explainer-backdrop");
        if (closeBtn) { closeBtn.addEventListener("click", closeExplainer); }
        if (backdrop) { backdrop.addEventListener("click", closeExplainer); }

        document.addEventListener("click", function (event) {
            var hint = event.target.closest && event.target.closest("details.hint");
            if (!hint) { return; }
            var summary = event.target.closest("summary");
            if (!summary) { return; }
            if (openExplainer(hint)) {
                event.preventDefault();
            }
        });

        document.addEventListener("keydown", function (event) {
            if (event.key === "Escape") { closeExplainer(); }
        });
    }

    function initHelp() {
        var hints = document.querySelectorAll("details.hint");
        if (!hints.length) { return; }

        Array.prototype.forEach.call(hints, function (hint) {
            hint.addEventListener("toggle", function () {
                if (hint.open) { closeHints(hint); }
            });
        });

        document.addEventListener("keydown", function (event) {
            if (event.key === "Escape") { closeHints(null); }
        });

        document.addEventListener("click", function (event) {
            if (!event.target.closest || !event.target.closest("details.hint")) {
                closeHints(null);
            }
        });

        initExplainer();
    }

    /* --- motion -----------------------------------------------------------

       Two effects, one observer. Sections rise into place as they are reached,
       and measured bars grow to a width the server rendered into a custom
       property. Both are one-shot: nothing loops, and nothing keeps moving
       after the page settles.

       The counters are the part that needs care. A figure on this site is
       evidence, so the animation interpolates toward the number but the final
       frame restores the server's exact string, character for character —
       never a re-formatted approximation of it. Anything with two numbers in
       it, or a non-numeric value, is left alone rather than parsed loosely.
       --------------------------------------------------------------------- */

    var COUNT_MS = 900;
    var COUNTABLE = ".stat-value.num, .kpi-value.num, .shift-fig.num, .bar-val.num";
    /* One leading number, then a suffix that contains no further digits. */
    var SINGLE_NUMBER = /^\s*([0-9][0-9,]*(?:\.[0-9]+)?)([^0-9]*)$/;

    function prefersReducedMotion() {
        return window.matchMedia
            && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    }

    function countUp(node) {
        if (node.dataset.counted === "1") { return; }
        node.dataset.counted = "1";

        var original = node.textContent;
        var match = SINGLE_NUMBER.exec(original);
        if (!match) { return; }

        var target = parseFloat(match[1].replace(/,/g, ""));
        if (!isFinite(target)) { return; }

        var decimals = (match[1].split(".")[1] || "").length;
        var grouped = match[1].indexOf(",") !== -1;
        var suffix = match[2];
        var started = null;

        function frame(now) {
            if (started === null) { started = now; }
            var t = Math.min(1, (now - started) / COUNT_MS);
            /* easeOutCubic: fast to begin, settling rather than stopping. */
            var eased = 1 - Math.pow(1 - t, 3);
            if (t < 1) {
                var value = target * eased;
                var text = decimals
                    ? value.toFixed(decimals)
                    : String(Math.round(value));
                if (grouped) {
                    text = Number(text).toLocaleString("en-US",
                        { minimumFractionDigits: decimals, maximumFractionDigits: decimals });
                }
                node.textContent = text + suffix;
                window.requestAnimationFrame(frame);
            } else {
                /* The committed string, restored verbatim. */
                node.textContent = original;
            }
        }
        window.requestAnimationFrame(frame);
    }

    function revealEverything() {
        Array.prototype.forEach.call(
            document.querySelectorAll(".reveal, .bar-fill"),
            function (n) { n.classList.add("is-in"); }
        );
    }

    function initMotion() {
        /* Opting in to the hidden state, rather than the stylesheet imposing it.
           Nothing above this line can leave the page invisible. */
        document.documentElement.classList.add("js-motion");

        var reveals = document.querySelectorAll(".reveal");
        var bars = document.querySelectorAll(".bar-fill");

        /* Failsafe. If the observer never fires — a viewport resized to the full
           document height by a screenshot tool, a browser reporting nothing as
           intersecting, anything unforeseen — the content appears regardless.
           Better to lose the animation than to lose the page. */
        window.setTimeout(revealEverything, 2500);

        /* Without IntersectionObserver, or with reduced motion requested,
           everything is placed in its final state immediately. The page must
           be complete without a single animation having run. */
        if (!("IntersectionObserver" in window) || prefersReducedMotion()) {
            Array.prototype.forEach.call(reveals, function (n) { n.classList.add("is-in"); });
            Array.prototype.forEach.call(bars, function (n) { n.classList.add("is-in"); });
            return;
        }

        var observer = new IntersectionObserver(function (entries) {
            entries.forEach(function (entry) {
                if (!entry.isIntersecting) { return; }
                var node = entry.target;
                node.classList.add("is-in");
                Array.prototype.forEach.call(node.querySelectorAll(COUNTABLE), countUp);
                if (node.classList.contains("bar-fill")) { countUp(node); }
                observer.unobserve(node);
            });
        }, { rootMargin: "0px 0px -8% 0px", threshold: 0.12 });

        Array.prototype.forEach.call(reveals, function (n) { observer.observe(n); });
        Array.prototype.forEach.call(bars, function (n) { observer.observe(n); });
    }

    function initNavState() {
        var head = byId("nav-head");
        if (!head) { return; }
        var update = function () {
            head.setAttribute("data-scrolled", window.scrollY > 8 ? "true" : "false");
        };
        update();
        window.addEventListener("scroll", update, { passive: true });
    }

    /* --- guided tour ------------------------------------------------------

       Fourteen steps across four routes, in the order the project should be
       read. Each step highlights something real on the page and explains it;
       the content is rendered by the server and read here as text, so the tour
       cannot say anything the page does not already contain.

       Two behaviours worth naming:

       - It runs ONCE per tab. The marker lives in sessionStorage, so moving
         between pages continues the sequence rather than restarting it, and
         closing the tab forgets it. The button in the header replays it on
         demand, which is what makes it useful in a demo rather than a nag.
       - A step whose target is missing is skipped, not fatal. Sections
         disappear when an artifact is absent, and a tour that breaks in that
         case would be worse than no tour.
       --------------------------------------------------------------------- */

    var TOUR_SEEN = "fsd.tour.seen";
    var TOUR_AT = "fsd.tour.at";
    var TOUR_RUNNING = "fsd.tour.running";

    var tour = { steps: [], pos: -1, nodes: null };

    function store(key, value) {
        try { window.sessionStorage.setItem(key, value); } catch (e) { /* private mode */ }
    }
    function recall(key) {
        try { return window.sessionStorage.getItem(key); } catch (e) { return null; }
    }
    function forget(key) {
        try { window.sessionStorage.removeItem(key); } catch (e) { /* no-op */ }
    }

    function readTourSteps() {
        var host = byId("tour-data");
        if (!host) { return []; }
        return Array.prototype.map.call(
            host.querySelectorAll(".tour-step-data"),
            function (node) {
                return {
                    index: Number(node.getAttribute("data-index")),
                    total: Number(node.getAttribute("data-total")),
                    target: node.getAttribute("data-target"),
                    title: node.querySelector(".tour-step-title").textContent,
                    body: node.querySelector(".tour-step-body").textContent
                };
            }
        );
    }

    var TOUR_ORDER = ["home", "metrics", "demo", "live"];
    var TOUR_HREF = { home: "/", metrics: "/metrics", demo: "/demo", live: "/live" };

    function currentPage() {
        return document.body.getAttribute("data-page") || "";
    }

    function nextPageAfter(page) {
        var i = TOUR_ORDER.indexOf(page);
        return (i > -1 && i < TOUR_ORDER.length - 1) ? TOUR_ORDER[i + 1] : null;
    }

    function tourChrome() {
        if (tour.nodes) { return tour.nodes; }

        var spot = el("div", { className: "tour-spot", id: "tour-spot" });
        var title = el("p", { className: "tour-title" });
        var body = el("p", { className: "tour-body" });
        var count = el("span", { className: "tour-count" });

        var prev = el("button", { className: "btn btn-quiet tour-prev", text: "Back", attrs: { type: "button" } });
        var next = el("button", { className: "btn btn-primary tour-next", text: "Next", attrs: { type: "button" } });
        var end = el("button", { className: "tour-end", text: "Skip tour", attrs: { type: "button" } });

        prev.addEventListener("click", function () { moveTour(-1); });
        next.addEventListener("click", function () { moveTour(1); });
        end.addEventListener("click", function () { endTour(true); });

        var card = el("div", { className: "tour-card", id: "tour-card", attrs: { role: "dialog", "aria-live": "polite" } }, [
            el("div", { className: "tour-card-hd" }, [count, end]),
            title, body,
            el("div", { className: "tour-actions" }, [prev, next])
        ]);

        var veil = el("div", { className: "tour-veil", id: "tour-veil" });
        veil.addEventListener("click", function () { endTour(true); });

        document.body.appendChild(veil);
        document.body.appendChild(spot);
        document.body.appendChild(card);
        tour.nodes = { spot: spot, card: card, veil: veil, title: title, body: body,
                       count: count, prev: prev, next: next };
        return tour.nodes;
    }

    function placeTour(target, step) {
        var n = tourChrome();
        var box = target.getBoundingClientRect();
        var pad = 8;
        var top = box.top + window.scrollY - pad;
        var left = box.left + window.scrollX - pad;

        n.spot.style.top = top + "px";
        n.spot.style.left = left + "px";
        n.spot.style.width = (box.width + pad * 2) + "px";
        n.spot.style.height = (box.height + pad * 2) + "px";

        n.title.textContent = step.title;
        n.body.textContent = step.body;
        n.count.textContent = "Step " + (step.index + 1) + " of " + step.total;
        n.prev.disabled = step.index === 0;
        n.next.textContent = step.index === step.total - 1 ? "Finish" : "Next";

        /* On a narrow screen the card is a bottom sheet and needs no placing;
           on a wide one it sits under the highlight, nudged back inside the
           viewport rather than allowed to hang off the edge. */
        if (window.matchMedia("(max-width: 760px)").matches) {
            n.card.style.top = "";
            n.card.style.left = "";
            return;
        }
        var cardW = 380;
        var cardTop = top + box.height + pad * 2 + 12;
        var cardLeft = Math.min(
            Math.max(12, left),
            Math.max(12, document.documentElement.clientWidth - cardW - 12)
        );
        n.card.style.top = cardTop + "px";
        n.card.style.left = cardLeft + "px";
    }

    function showTourStep(pos) {
        var step = tour.steps[pos];
        if (!step) { return false; }
        var target = document.querySelector(step.target);
        if (!target || !target.getBoundingClientRect().height) { return false; }

        tour.pos = pos;
        store(TOUR_AT, String(step.index));
        document.body.classList.add("tour-open");

        /* The highlight is positioned in DOCUMENT coordinates, so it is placed
           immediately and correctly regardless of where the page is scrolled.
           It is placed a second time once the scroll and any reveal transitions
           have settled, because those can change an element's height under it. */
        placeTour(target, step);
        target.scrollIntoView({
            block: "center",
            behavior: prefersReducedMotion() ? "auto" : "smooth"
        });
        window.setTimeout(function () { placeTour(target, step); }, 380);
        window.setTimeout(function () { placeTour(target, step); }, 800);
        return true;
    }

    function moveTour(delta) {
        var pos = tour.pos + delta;
        while (pos >= 0 && pos < tour.steps.length) {
            if (showTourStep(pos)) { return; }
            pos += delta;
        }
        if (delta > 0) {
            var next = nextPageAfter(currentPage());
            if (next) {
                store(TOUR_RUNNING, "1");
                window.location.href = TOUR_HREF[next];
                return;
            }
            endTour(true);
            return;
        }
        endTour(true);
    }

    function endTour(markSeen) {
        document.body.classList.remove("tour-open");
        if (tour.nodes) {
            [tour.nodes.veil, tour.nodes.spot, tour.nodes.card].forEach(function (n) {
                if (n && n.parentNode) { n.parentNode.removeChild(n); }
            });
            tour.nodes = null;
        }
        tour.pos = -1;
        forget(TOUR_RUNNING);
        if (markSeen) { store(TOUR_SEEN, "1"); }
    }

    function startTour(fromStart) {
        tour.steps = readTourSteps();
        if (!tour.steps.length) {
            var next = nextPageAfter(currentPage());
            if (next) { store(TOUR_RUNNING, "1"); window.location.href = TOUR_HREF[next]; }
            return;
        }
        var at = fromStart ? -1 : Number(recall(TOUR_AT) || -1);
        var pos = 0;
        for (var i = 0; i < tour.steps.length; i += 1) {
            if (tour.steps[i].index >= at) { pos = i; break; }
        }
        store(TOUR_RUNNING, "1");
        if (!showTourStep(pos)) { moveTour(1); }
    }

    function initTour() {
        var button = byId("tour-start");
        if (button) {
            button.addEventListener("click", function () {
                /* Replay means the whole sequence, from step one. Step one
                   lives on the overview, so from anywhere else the button goes
                   there first rather than starting halfway through. */
                forget(TOUR_AT);
                forget(TOUR_SEEN);
                endTour(false);
                if (currentPage() !== TOUR_ORDER[0]) {
                    store(TOUR_RUNNING, "1");
                    window.location.href = TOUR_HREF[TOUR_ORDER[0]];
                    return;
                }
                startTour(true);
            });
        }
        if (!byId("tour-data")) { return; }

        document.addEventListener("keydown", function (event) {
            if (!document.body.classList.contains("tour-open")) { return; }
            if (event.key === "Escape") { endTour(true); }
            if (event.key === "ArrowRight") { moveTour(1); }
            if (event.key === "ArrowLeft") { moveTour(-1); }
        });

        window.addEventListener("resize", function () {
            if (!document.body.classList.contains("tour-open")) { return; }
            var step = tour.steps[tour.pos];
            if (!step) { return; }
            var target = document.querySelector(step.target);
            if (target) { placeTour(target, step); }
        });

        /* Mid-tour on a new page: continue. Otherwise offer it once per tab. */
        if (recall(TOUR_RUNNING) === "1") {
            window.setTimeout(function () { startTour(false); }, 400);
        } else if (recall(TOUR_SEEN) !== "1") {
            window.setTimeout(function () { startTour(true); }, 900);
        }
    }

    /* --- console shell: off-canvas sidebar on narrow viewports ------------- */
    function initShell() {
        var toggle = byId("shell-toggle");
        var side = byId("shell-side");
        var backdrop = byId("shell-backdrop");
        if (!toggle || !side || !backdrop) { return; }

        var close = function () {
            side.classList.remove("is-open");
            backdrop.classList.remove("is-open");
            backdrop.hidden = true;
            toggle.setAttribute("aria-expanded", "false");
        };
        var open = function () {
            side.classList.add("is-open");
            backdrop.classList.add("is-open");
            backdrop.hidden = false;
            toggle.setAttribute("aria-expanded", "true");
        };

        toggle.addEventListener("click", function () {
            if (side.classList.contains("is-open")) { close(); } else { open(); }
        });
        backdrop.addEventListener("click", close);
        document.addEventListener("keydown", function (event) {
            if (event.key === "Escape") { close(); }
        });
        Array.prototype.forEach.call(side.querySelectorAll(".shell-nav-link"), function (link) {
            link.addEventListener("click", close);
        });
    }

    /* --- page setup -------------------------------------------------------

       One script serves four pages, so every block is guarded on the elements
       it needs rather than on a page name. A surface can move between routes
       without this file having to be told about it.
       --------------------------------------------------------------------- */

    initHelp();
    initMotion();
    initNavState();
    initShell();
    initTour();

    Array.prototype.forEach.call(document.querySelectorAll(".tab"), function (tab) {
        tab.addEventListener("click", function () { showView(tab.dataset.view); });
    });

    var liveButton = byId("btn-live-trigger");
    if (liveButton) {
        liveButton.addEventListener("click", triggerLiveTransaction);
    }

    initScorer();

    if (byId("pending-container")) {
        /* Arriving from another page's "view the audit trail" link. */
        var hash = window.location.hash || "";
        var wantsAudit = hash.indexOf("#audit") === 0;
        showView(wantsAudit ? "audit" : "pending");
        load().then(function () {
            if (!wantsAudit) { return; }
            var flashId = hash.split("-")[1];
            showAuditTrail(flashId === undefined ? undefined : Number(flashId));
        });
    } else if (byId("live-items-container")) {
        loadLiveItems();
    }
})();
