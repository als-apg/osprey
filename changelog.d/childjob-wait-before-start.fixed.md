`ChildJob.wait()` on a job that was neither started nor stopped now raises `RuntimeError` naming the job, instead of ending it as cancelled and leaving a later `start()` to spawn nothing.
