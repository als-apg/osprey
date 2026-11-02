"""An OSPREY application: one panel over one channel, one read tool, one write tool."""

#: The application's name, which is also its panel id and its tool server's name.
APP_NAME = "knob-example"
#: The backend routes the tool server reaches through the panel door.
STATE_ROUTE = "/api/state"
WRITE_ROUTE = "/api/write"
