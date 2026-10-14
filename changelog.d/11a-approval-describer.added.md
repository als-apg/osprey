The approval prompt of a read-write `execute` or `execute_file` call binds it to
its control target and asks: the approved call carries `approved_target`, set
over any value the agent supplied. A read-only call keeps its prompt and carries
no binding. A run on a held target fails before it starts, naming the holder.
