---
name: nginx-triage
description: Work out why nginx is failing on a host, in order of cheapest evidence
when: a healthcheck reports nginx down and you have not read any log yet
args: host
risk: read
run: ssh {host} "systemctl status nginx --no-pager; journalctl -u nginx -n 120 --no-pager; nginx -t 2>&1"
---

# nginx triage

Do these in order, and stop as soon as the cause is obvious:

1. `systemctl status nginx` — is the unit failing to start, or running but misbehaving?
   `Active: failed` with an exit code means config or bind failure, not load.
2. `journalctl -u nginx -n 120` — look for the **first** error, not the last. Later ones are
   usually the restart loop repeating the same failure.
3. `nginx -t` — configuration syntax and cert paths. This is read-only and safe.

Decision table:

| symptom | likely cause | next action |
|---|---|---|
| `bind() to 0.0.0.0:443 failed (98)` | another process holds the port | `ss -ltnp \| grep :443` |
| `SSL_CTX_use_PrivateKey_file ... failed` | cert/key mismatch or wrong path | verify files, do not guess paths |
| `worker_connections are not enough` | limit too low for real traffic | raise it, then reload |
| `connection refused` from the healthcheck only | healthcheck points at the wrong upstream | compare URLs before touching nginx |

Rules:

* Prefer `reload` over `restart`; a restart drops in-flight connections.
* Never edit a config file you have not read in full.
* If the fix is `rm`, `chmod -R`, or a firewall flush, stop and ask a human.
