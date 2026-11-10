"""An OSPREY application: one panel over one channel, one read tool, one write tool."""

#: The application's name, which is also its panel id and its tool server's name.
APP_NAME = "scaffold-app"
#: The backend routes the tool server reaches through the panel door.
STATE_ROUTE = "/api/state"
WRITE_ROUTE = "/api/write"
#: The key of the application's config naming the channel it shows and sets.
CHANNEL_KEY = "channel"
#: That key as the deployment's profile spells it.
CHANNEL_SETTING = f"apps.{APP_NAME}.config.{CHANNEL_KEY}"
