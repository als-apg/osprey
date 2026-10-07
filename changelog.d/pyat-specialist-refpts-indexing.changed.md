The pyAT specialist's recipe now says how `get_optics` rows map to elements:
one row per `refpts` entry, in that order, so a value at a named element is
read by the element's position within the `refpts` passed, which equals its
ring index only when every element was requested.
