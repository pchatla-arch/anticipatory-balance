# Foresight technical notes

Reference material for the numerical, perception, decision, and control choices in Foresight.
Each note distinguishes observed evidence from reasoning and proposed experiments.

| Note | Questions covered | Evidence status | Discussion |
| --- | --- | --- | --- |
| [Reliable humanoid execution roadmap](humanoid_execution_roadmap.md) | How does actuator adaptation support useful humanoid work? Which task, RL comparison, runtime, and perception evidence is needed? | Proposed milestones grounded in public robotics engineering requirements; current gate remains unchanged. | Portfolio branch `portfolio/adaptive-execution-gate` |
| [Adaptive execution and the actuator feasibility gate](adaptive_execution.md) | What adapts while PPO stays frozen? How does this differ from bracing? What does the known-gain gate prove? | Implemented with local CPU checks; no GPU recovery or online adaptation result yet. | Portfolio branch `portfolio/adaptive-execution-gate` |
| [Controllers, frozen skills, and bracing decisions](controllers_and_freezing.md) | Which controllers exist? What is frozen? How does stillness differ from BRACE? Why vary decision timing separately from skill training? | Source and checkpoint audit at `67abf97`; implemented, currently evaluated, and proposed methods are distinguished. | [Issue #7](https://github.com/pchatla-arch/anticipatory-balance/issues/7) |
| [Numerical precision in Foresight](numerical_precision.md) | Why did the results audit reject a tiny rounding difference? What changes with FP16, BF16, INT8, or FP8? How should lower-precision controllers be evaluated? | The FP32/FP64 reporting discrepancy was reproduced and corrected. Reduced-precision controller comparisons are proposed experiments, not measured results. | [Issue #6](https://github.com/pchatla-arch/anticipatory-balance/issues/6) |

Use each note's discussion thread for questions, follow-up decisions, and links to future
benchmark evidence. Neither note introduces new GPU measurements or declares Session H complete.

The main project is described in the [repository README](../../README.md).

## Maintaining these notes

Record the reviewed source revision and link to the original experiment evidence.
Keep hypothetical calculations labeled as examples. Add benchmark outcomes only when
their code, precision settings, evaluation scenarios, and measurement method are recorded.
