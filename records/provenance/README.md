# Provenance of the repository's history

`github_activity_2026-10-10.json` is the hosting service's record of this repository's
activity (GitHub's repository activity endpoint), exported on 10 October 2026 at
09:50 UTC with

    gh api -X GET repos/HymnOfLight/FE-JEPA/activity -f per_page=100 -f direction=asc --paginate

The pages were joined into one JSON array, sorted by timestamp and written with one-space
indentation; the events are as returned. It holds 103 events, from the creation of `main`
on 14 July 2026 to the push to `main` at 09:28 UTC on 10 October 2026: 96 pushes,
5 branch creations, 1 force push and 1 deletion (of the tag `prereg-phase2b`, on
22 September 2026). It logs no tag creations, and it does not show whether the repository
was public at a given time. Before 6 August 2026 it logs two events, both on `main`:

| Time (UTC) | Event | Before | After |
|---|---|---|---|
| 2026-07-14T03:25:10Z | branch creation | (none) | `b365af58b5135c6d079b377c26af419bdfe5fcaa` |
| 2026-08-02T17:29:37Z | force push | `b365af58b5135c6d079b377c26af419bdfe5fcaa` | `a548825da6d02c91c7ccc4b6fcd6814116b2117e` |

`b365af5` ("Add files via upload", 14 July 2026 at 03:25 UTC) was the first commit of the
repository's first history, and the only one. It was made through the hosting service's web
interface, which created it (committer GitHub) and signed it; the service reports the
signature as verified, and its packet names GitHub's web-flow key `B5690EEEBB952194` and
the time 03:25:10 UTC. It held:

- the code of release v2.1.4, which differs from v2.1.5 (`a548825`) only in the files that
  `FIX_NOTES_v2_1_5.md` lists (`src/fejepa/__init__.py`, `src/fejepa/experiments/protocol.py`,
  `src/fejepa/experiments/runner.py`, the new `tests/test_asis_guard.py`);
- `PREREG.md` as shipped with that release, a template whose `CONFIG_SHA256` line was blank
  (`<fill before tagging>`): the only difference from the stamped document that the run of
  16 July 2026 checked (SHA-256 `b9470e68...`, as `PROVENANCE_NOTE.md` records);
- `configs/phase1_rec8_v2.json`, byte for byte as in this repository (canonical SHA-256
  `62b26ad868d424ef5527c8cb7d826c818aa1ba5cebbc76c7bfe665062781f0ce`, the stamped hash).

It did not hold `PREREG_WP2.md`. The force push replaced that history with the one that
begins at the root commit `a548825` ("v2.1.5", dated 2026-08-03 01:26 +0800, that is,
2 August 2026 at 17:26 UTC), which holds both stamped documents and `PROVENANCE_NOTE.md` as
compiled on 1 August 2026; the note records their SHA-256 hashes (`b9470e68...` and
`550cf7c6...`). The tag `provenance-2026-07-14`, created in October 2026, keeps `b365af5`.
See also the addendum of 10 October 2026 in `PROVENANCE_NOTE.md`.

## The owner account's security log

`security_log_FE-JEPA_2026-10-10.json` holds the 15 entries that name `FE-JEPA` in the
security log of the repository's owner account, `HymnOfLight`, as exported by the account
holder on 10 October 2026 (file `export-HymnOfLight-1791628693.json.gz`, SHA-256
`980044307e7f171aa4f92cc00658f781370e36cf7e4597d483a57eddc67cd042`; 643 entries, from
13 April to 10 October 2026). Each entry keeps the fields `@timestamp`, `_document_id`,
`action`, `actor`, `actor_id`, `created_at`, `operation_type`, `repo`, `repo_id`,
`visibility`, `public_repo`, `user`, `user_id`, `invitee`, `inviter`, `integration`,
`repositories_added_names` and `repositories_removed_names` where present; user agents,
request identifiers and request headers are removed, and no other entry of the export is
committed. Times below are `@timestamp` (milliseconds since 1970, UTC) converted.

| Time (UTC) | Action | Repository id | `public_repo` |
|---|---|---|---|
| 2026-06-13T08:36:47.792Z | `repo.create` | 1268189576 | true |
| 2026-07-14T03:23:17.268Z | `repo.destroy` | 1268189576 | true |
| 2026-07-14T03:23:30.468Z | `repo.create` | 1299960094 (the present repository) | true |
| 2026-08-06T14:47:18.032Z | `repository_invitation.create` (invitee `RuifengCao`) | 1299960094 | true |

The other entries are the settings and integrations that the service records with each
creation and deletion; every entry that has `visibility` reads `public`. The present
repository was thus created public, 13 seconds after a
public repository of the same name, created on 13 June 2026, had been deleted, and about
95 seconds before `b365af5`. No entry of the whole export has the action `repo.access`,
which GitHub's documentation describes as "The visibility of a repository changed": the
visibility of none of the account's repositories changed in that period.

| File | SHA-256 |
|---|---|
| `github_activity_2026-10-10.json` | `f78f0071e040bfc56f87132508148df039f57195efae83b671f19ee11f2907bc` |
| `security_log_FE-JEPA_2026-10-10.json` | `dcb0a879ee700d21897be2e10c2f5585ce85202684d453286d4e6d1607c6bf7f` |
