ARIEL search-time provider calls (query embedding and the reachability check
behind it) work in a process forked after the parent has made one: each
process now admits these calls through its own bound, so a child never waits on
a worker that exists only in its parent. A call the search gave up on after its
timeout keeps running on a daemon thread, so it no longer keeps the interpreter
alive at exit while the provider has not answered.
