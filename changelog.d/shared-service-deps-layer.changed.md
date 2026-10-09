Service images now share one framework install. Every Python service recipe
is identical through its deps layer, so `osprey up` installs the framework once
per deploy rather than once per image, and each image adds only what it needs
(an extra, the queueserver pin, Node) in a layer after it.
