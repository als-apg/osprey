`osprey scaffold web-terminals lint` and `osprey up` name a roster user whose
stored password hash in `.env.auth` the login service cannot evaluate, so the
broken entry is found before anyone tries to log in.
