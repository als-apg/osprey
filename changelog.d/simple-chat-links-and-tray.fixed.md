In the Simple view, a link in an agent answer to the loopback address of a
panel server now opens that panel in the workspace. An example is the gallery
address `http://127.0.0.1:<port>#focus` that the artifact tools give to the
agent. Before, the link opened a new tab. That tab did not connect when the
operator used the hub from a different computer, for example through an SSH
tunnel. A loopback link to a different port still opens a new tab. The
`/api/panels` response now gives the port of each panel server that uses a
loopback address (`loopback_ports`). It gives no other data about the server.

The Simple view no longer removes the files in the attachment tray when it
reads the same conversation again. This occurs after a feed reconnect, at the
end of a turn from a different tab, and after a hand-off. Before, the tray
became empty and the server deleted the files that the operator attached but
did not send. A new conversation and a change to a different conversation
still empty the tray and delete those files.
