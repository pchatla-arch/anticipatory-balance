# AE-013 · A skipped sample must preserve a bounded gain interval

On the untouched WP 2.2 commit `ffa76a677046dbab7aeb244ed93901e838125d8b`, the CPU test `test_padding_and_saturation_preserve_evidence_and_clock` expects the estimated interval’s lower endpoint to equal the true gain 0.70 to four decimal places. The exact sensor preset still declares nonzero encoder and torque bounds. Valid samples at gain 0.70 consequently produce a lower endpoint of 0.6997153162956238 using JAX 0.12.0.1.

The intended regression is that an unreleased packet and a saturated packet retain the existing interval when the configured gain drift is zero. Check both endpoints against a snapshot before those packets and check that the original interval contains the true gain. Preserve the independent source-clock and informative-count assertions.
