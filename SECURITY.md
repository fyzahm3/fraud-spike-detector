# Security

## Reporting a vulnerability

Please open a [private security advisory](https://github.com/fyzahm3/fraud-spike-detector/security/advisories/new)
rather than a public issue. Include what you found, how to reproduce it, and what an attacker gains.
You will get a response within a few days.

## Scope and design

This is a research and demonstration project, not production payment infrastructure. It is worth
knowing what it deliberately does and does not do.

**It cannot move money.** There is no blocking, holding, cancelling, refunding, capturing or payout
code anywhere in the repository. The only outbound payment-rail call is a single test-mode order
creation. Automated tests scan the source for action-capable terms on every run.

**Test mode is enforced, not assumed.** A Razorpay key id that does not carry the `rzp_test_` prefix
is refused during credential loading, before any socket is opened.

**Webhooks are verified before they are parsed.** HMAC-SHA256 over the raw request body against the
signing secret. An unverified payload is rejected with a 400 and is never parsed, stored, or logged.
Replayed event ids are rejected by an atomic insert, so a retried delivery cannot enqueue twice.

**Reading is public; changing state is not.** The auth gate keys off the HTTP method rather than a
path allowlist, so a new route is public only while it stays read-only and is protected automatically
once it accepts a POST. This fails closed.

**Mutations require a CSRF double-submit pair**, and `POST /api/resolve` never defaults its `action` —
a resolution recorded without an explicit human choice would put a decision nobody made into an
append-only log.

**The dashboard escapes structurally.** Queue data — including LLM-generated prose — reaches the page
through `textContent` on constructed DOM nodes. Nothing is assigned to `innerHTML`, and tests scan
every UI source file to keep it that way.

## Secrets

No credential is ever committed. `.env` is ignored, `.env.example` carries empty placeholders, and a
secret-scan pre-commit hook lives at `scripts/hooks/pre-commit` — enable it with
`git config core.hooksPath scripts/hooks`.

Dependencies are audited with `pip-audit -r requirements.txt`.
