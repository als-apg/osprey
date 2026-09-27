The event dispatcher records who fired each run. A fire or re-fire made
from the web terminal stores the person's name as `owner` on the trigger's
history entry and on the worker's run record; a webhook, cron or chat-bridge
fire is stored without one. The dashboard's run detail shows it in Expert
mode.
