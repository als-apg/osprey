Execution metadata written by an agent Python run now carries a UTC offset on
its `start_time` and `end_time`, matching every other timestamp OSPREY reports
and making the records orderable against them. The executor timeout also has
one default: the value the config builder falls back to and the value the
executor falls back to are now the same constant.
