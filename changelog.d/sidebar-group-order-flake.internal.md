The sidebar group heading browser tests, and the context-menu test that opens a
heading's menu from the keyboard, wait for the rail's JavaScript render (shown
by the System group it adds) before they read the group order or open a
heading's menu, so a slow CI runner no longer reads a half-built sidebar. The
panel browser tests that hide or show a panel through the server give its
broadcast the same 10 s budget as the file's other server round trips.
