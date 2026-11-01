A profile that still sets `web.bar_items` builds, with one warning: the web
terminal's sidebar replaced the header and the status bar that key arranged,
so the key is dropped at the parse point and ignored rather than refused.
The warning names the sidebar and says to drop the key; `osprey init
--revalidate` rewrites the profile.
