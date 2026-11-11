The sidebar group heading browser tests, and the context-menu test that opens a
heading's menu from the keyboard, wait for the rail's JavaScript render (shown
by the System group it adds) before they read the group order or open a
heading's menu, so a slow CI runner no longer reads a half-built sidebar.
