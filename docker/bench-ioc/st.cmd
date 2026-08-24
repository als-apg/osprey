# Bench IOC startup script, run by softIoc -S (see the Containerfile CMD).
#
# Order matters and is not stylistic: asSetFilename only records where the
# access-security file lives, and iocInit is what actually reads it. Naming the
# file after iocInit leaves every record on the default group with full write
# access, so the one channel this IOC is supposed to refuse writes on would
# quietly accept them -- a green test proving the opposite of what it claims.
asSetFilename("/bench/bench.acf")

# Absolute paths throughout: softIoc resolves relative paths against its working
# directory, and the image sets none.
dbLoadRecords("/bench/bench.db")

# Prints "iocRun: All initialization complete" when the CA server is listening
# and every record is live. That line is the readiness marker callers poll for.
iocInit
