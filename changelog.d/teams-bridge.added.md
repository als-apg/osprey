A Microsoft Teams bridge joins the Google Chat and Nextcloud Talk ones:
mention the bot in a team channel or a group chat, or write to it in a 1:1
chat, and the OSPREY agent answers in the same conversation, with its PNG
plots inline. `teams_bridge:` in a build profile deploys it, and the adapter
installs from a new `teams` extra. A Teams bot receives messages only by
HTTPS POST, so the build also writes an Azure Function relay that checks each
request's Bot Framework token and puts the activity on a Service Bus queue;
the bridge pulls that queue, so it opens no port of its own and a question
asked while it is down waits on the queue until it comes back. `TEAMS_CLOUD`
picks the endpoints: `commercial` by default, `gcchigh` for a GCC High
tenant. The acknowledgement names the OSPREY version from
`APP_VERSION_DISPLAY`, falling back to the installed `osprey-framework`
version when that variable is unset.
