A channel bridge now announces a delayed answer. When a question was parked in
the retry queue during an outage and its retry finally completes, the answer
opens with a "Resuming your request queued at <time> (<elapsed> ago) — service
is restored, answer follows" line in the same thread, so an answer that lands
hours after the "queued" notice says which question it belongs to. Every
bundled channel (Google Chat, Microsoft Teams, Nextcloud Talk) posts it; a
custom `ChannelOps` adds the new `post_resumed` member.
