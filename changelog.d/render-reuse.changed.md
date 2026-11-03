A facility build writes each facility view once per distinct set of config
keys it reads and copies it into every other render, and the approval hook's
measurement block is computed once per distinct input instead of per render.
