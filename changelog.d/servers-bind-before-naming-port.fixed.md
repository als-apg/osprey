The translation proxy, `osprey web` and `osprey theme-lab` bind their
listening socket before they name a port, and serve on that socket, so the
port they report can no longer be taken by another process in between.
