A panel whose config request fails or does not get an answer at page load no
longer stays disabled on the rail until the next reload. The web terminal now
stops each config request after 5 seconds and tries again with backoff, up to
every 15 seconds, until it gets an answer. A request can be queued behind the
browser's limit of six connections to one host, so before this change one slow
load could disable every rail entry at once. A panel page that has not loaded
after 15 seconds now shows the "Backend not answering" notice on its tile
instead of a blank rectangle.
