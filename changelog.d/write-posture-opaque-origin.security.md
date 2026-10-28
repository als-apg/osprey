**Breaking change:** content an agent can write, such as artifacts, gallery
files and channel-finder output, is no longer served on the console's origin.
It renders as before in an opaque origin, but can no longer call OSPREY's own
APIs, so it cannot switch writes on.
