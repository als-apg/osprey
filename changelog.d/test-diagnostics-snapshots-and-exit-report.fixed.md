A CI container snapshot stops at its first timed-out engine call, runs outside the
failing test's timeout and says `capture incomplete` when it could not finish; the
end-of-run thread report leaves idle executor workers out of its count; and a test
session fails when a test writes into the packaged template tree.
