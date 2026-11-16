# Issue notes

Every problem found in either project gets a technical note here **and** a GitHub Issue,
linked both ways. The note is the durable engineering record: what went wrong, why, and
what changes. The Issue is the tracker: status, owner, discussion.

## Rules

1. **Open a note when a problem is found**, before the fix, from the template. Use the next
   ID in its series: `AE-nnn` (adaptive execution) or `FS-nnn` (Foresight).
2. **Evidence first.** Mark each claim *verified* (with a command or result file that
   reproduces it) or *hypothesis* (with the check that would settle it).
3. **Say why it did not work**, not only what failed: the mechanism, and why it was not
   caught earlier.
4. **State the next step or design change**, with the owner and the acceptance test.
5. **Close the note when the Issue closes**: fill in *Resolution* with the commit, PR or run
   that settled it. A rejected or superseded design stays in the record with its reason.
6. Notes are append-only in substance: correct a mistake with a dated line, do not delete it.
7. Any PR that fixes an issue cites its ID in the title; the reviewer checks that the note was
   updated.

## Index

| ID | Title | Severity | Status | Issue | Owner |
| --- | --- | --- | --- | --- | --- |
| [AE-001](AE-001-effort-state-alignment.md) | Effort sample paired with the wrong joint position | Blocking (Stage B) | Open | [#9](https://github.com/pchatla-arch/anticipatory-balance/issues/9) | ChatGPT (telemetry), Claude reviews |
| [AE-002](AE-002-pre-limit-actuator-force.md) | actuator_force is read before joint torque limits | Major | Open | [#10](https://github.com/pchatla-arch/anticipatory-balance/issues/10) | ChatGPT (telemetry), Claude reviews |
| [AE-003](AE-003-confounds-per-telemetry.md) | Stage B confounds cannot fail under effort telemetry | Major | Open | [#11](https://github.com/pchatla-arch/anticipatory-balance/issues/11) | Claude (Stage B), ChatGPT reviews |
| [AE-004](AE-004-no-effort-arm.md) | No-effort identification arm is unspecified | Major | Open | [#12](https://github.com/pchatla-arch/anticipatory-balance/issues/12) | Claude, ChatGPT reviews |
| [AE-005](AE-005-ankle-roll-identifiability.md) | Ankle-roll gain is weakly identifiable | Minor | Open | [#13](https://github.com/pchatla-arch/anticipatory-balance/issues/13) | Claude |
| [AE-006](AE-006-gate-degradation-floor.md) | Feasibility ratio can rest on a small denominator | Minor | Open | [#14](https://github.com/pchatla-arch/anticipatory-balance/issues/14) | ChatGPT (gate) |
| [FS-001](FS-001-session-h-nonfinite-progress.md) | Session H hold ablation stopped on non-finite progress | Major (Foresight) | Open | [#15](https://github.com/pchatla-arch/anticipatory-balance/issues/15) | ChatGPT (Session H), Claude reviews |
