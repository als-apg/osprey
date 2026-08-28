``osprey up``, the render step and the new ``web_terminals.open_mode_egress``
lint rule refuse an open deployment whose personas can still reach the host
network, naming the offending persona and the deny entries it is missing. Under
that posture the python executor also refuses connections from executed code to
the deployment's own web ports.
