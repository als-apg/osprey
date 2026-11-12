The virtual accelerator's periodic pass posts only the channels whose value
changed since their last post, so a put no longer waits behind a post of every
channel. An unchanged channel keeps the timestamp of its last post.
