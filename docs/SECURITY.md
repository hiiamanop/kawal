# Security Policy

This document defines the minimum security standard for code, infrastructure, dependencies, CI/CD, integrations, and AI-assisted features in this repository.

Security is a delivery requirement, not a final review step. If a feature cannot meet a REQUIRED control, do not release it until an approved, time-bound exception exists.

## 1. Reporting a Vulnerability

Do not report suspected vulnerabilities through a public issue, public pull request, or social-media post.

Report privately to the repository owner or the designated security contact. Include, where safe to share:

- affected version, commit, environment, or component;
- reproducible steps or a minimal proof of concept;
- expected and observed behavior;
- likely impact and any required attack preconditions;
- logs or screenshots with passwords, access tokens, personal data, and private URLs removed; and
- suggested mitigation, if known.

The team will acknowledge, triage, reproduce, contain, remediate, and validate the report. Public disclosure is coordinated only after a fix or mitigation is available. There is no bug bounty unless one is explicitly announced.

Researchers must not access other users' data, disrupt services, bypass access controls, or perform destructive testing.

## 2. Supported Versions

Until a formal release policy exists, the active default branch and the latest released version are the supported targets for security fixes. Older versions are handled on a best-effort basis.

| Version / branch | Security support |
|---|---|
| Active default branch | Supported |
| Latest release | Supported |
| Earlier releases | Best effort |

## 3. Required Security Controls

### 3.1 Secrets and credentials

- **REQUIRED:** Never commit secrets, private keys, API tokens, database passwords, session files, or production data.
- **REQUIRED:** Keep secrets in environment variables or an approved secret manager; use separate credentials per environment.
- **REQUIRED:** Rotate and revoke exposed credentials immediately, then audit relevant access.
- **REQUIRED:** Prevent secrets from entering client bundles, logs, error messages, fixtures, screenshots, and documentation.
- **SHOULD:** Use short-lived credentials and MFA for privileged accounts where the platform supports them.

### 3.2 Authentication and authorization

- **REQUIRED:** Enforce authorization server-side for every resource, tenant, and privileged action. Client-side hiding is not authorization.
- **REQUIRED:** Apply deny-by-default and least privilege to users, services, database roles, storage, CI, and external tools.
- **REQUIRED:** Use vetted authentication providers. If passwords are stored, hash them using Argon2id or bcrypt; never store reversible passwords.
- **REQUIRED:** Protect cookie sessions with `HttpOnly`, `Secure`, and appropriate `SameSite` attributes; add CSRF protection to state-changing cookie-authenticated requests.
- **REQUIRED:** Rate-limit and monitor authentication, password reset, public form, and expensive API endpoints.

### 3.3 Input, output, and file handling

- **REQUIRED:** Treat all requests, webhooks, uploads, URLs, model outputs, and third-party responses as untrusted.
- **REQUIRED:** Validate input at each trust boundary using explicit schemas, allowlists, size limits, and normalized formats.
- **REQUIRED:** Use parameterized queries or a safe ORM; never interpolate untrusted values into SQL, shell commands, HTML, templates, or paths.
- **REQUIRED:** Encode output for its target context and use framework-native protections against XSS and CSRF.
- **REQUIRED:** Validate file type using content and MIME checks, set size limits, store uploads privately by default, and do not execute uploaded content.
- **REQUIRED:** Image egress to an external model requires an explicit `ALLOW` from a deterministic non-vision policy gate using permitted metadata and text signals. Fail closed on sensitivity or ambiguity; retain the image privately and obtain needed facts through text clarification.
- **REQUIRED:** Fetch user-provided URLs only through an SSRF-safe allowlist/proxy policy; block local, metadata, and private network targets.

### 3.4 Data protection and privacy

- **REQUIRED:** Collect, retain, and expose only the data needed for the stated feature.
- **REQUIRED:** Encrypt data in transit using TLS and enable encryption at rest using supported platform controls.
- **REQUIRED:** Redact personal data, secrets, authentication headers, and full sensitive payloads from telemetry.
- **REQUIRED:** Define ownership, retention, deletion, export, and access rules before processing production personal data.
- **REQUIRED:** Separate production, staging, development, and test data. Production data may not be copied to lower environments without written authorization and effective protection.

### 3.5 Dependencies and supply chain

- **REQUIRED:** Commit lockfiles and install dependencies from trusted registries.
- **REQUIRED:** Review new dependencies, GitHub Actions, container images, and build scripts before use.
- **REQUIRED:** Run dependency and secret scanning in CI where supported; prioritize remediation of actively exploitable critical issues.
- **REQUIRED:** Pin CI actions and production images to reviewed versions or immutable digests when practical.
- **REQUIRED:** Do not run untrusted scripts, install commands copied from unknown sources, or build artifacts with undisclosed credentials.

### 3.6 Secure delivery and operations

- **REQUIRED:** Review security-sensitive changes, including authentication, authorization, payments, data exports, infrastructure, CI, and external integrations.
- **REQUIRED:** Keep debug endpoints, test credentials, broad CORS rules, and development bypasses disabled outside development.
- **REQUIRED:** Use protected environment configuration, health checks, backups where applicable, and a tested rollback path before production deployment.
- **REQUIRED:** Log authentication events, authorization failures, privileged actions, configuration changes, security-policy denials, and external side effects with request/correlation IDs.
- **REQUIRED:** Restrict access to logs and audit trails. Logs must not contain passwords, tokens, full payment data, or unrestricted sensitive payloads.

## 4. AI and Agent Safety

For systems that call models, tools, or external services:

- **REQUIRED:** Treat prompts, retrieved data, attachments, and model output as untrusted input.
- **REQUIRED:** Use structured output validation before model output affects state, authorization, routing, or external actions.
- **REQUIRED:** Give agents only scoped credentials and explicit tool allowlists; agents must not obtain arbitrary shell, network, database, or secret access.
- **REQUIRED:** Place deterministic policy and authorization checks before destructive, financial, data-disclosing, or external actions.
- **REQUIRED:** Do not use a local vision model or model-based image redaction in P0. Remote vision must be gated by deterministic non-vision signals and fail closed for sensitive or ambiguous images.
- **REQUIRED:** Do not send raw secrets or sensitive personal data to third-party model providers without an approved data-handling policy.
- **REQUIRED:** Record model, prompt, tool, policy, and side-effect provenance sufficient to investigate an incident without storing unnecessary raw content.

## 5. Incident Response

When a security incident is suspected:

1. Contain the issue: disable affected credentials, integration, feature, or endpoint when necessary.
2. Preserve the minimum forensic evidence needed for investigation without increasing exposure.
3. Assess affected systems, data, users, and continuing risk.
4. Remediate the vulnerability, rotate relevant secrets, test the fix, and deploy through the normal controlled path.
5. Communicate through approved channels and follow applicable notification obligations.
6. Write a post-incident review with root cause, impact, timeline, corrective actions, and owners.

Do not delete logs or alter evidence to conceal an incident.

## 6. Security Exceptions

An exception must be documented before release and include:

- the exact control and affected scope;
- owner and approver;
- risk rationale and compensating controls;
- start date, expiry/review date, and removal plan.

Expired exceptions are not valid. An exception never permits bypassing law, platform controls, access controls, or a confirmed critical vulnerability without explicit emergency authorization.

## 7. Security Update Practice

Material security fixes should include release notes or an advisory appropriate to the affected users, plus upgrade, configuration, credential-rotation, or mitigation instructions when relevant. Give reporters credit only with their consent.
