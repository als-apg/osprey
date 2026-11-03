Shared containers (the Bluesky queue, the dispatch worker) read a
build-rendered login→tier map, `control_system.login_tiers`, and write only
when the container's tier, the owner's tier and the owner's switch all allow
it. A login missing from the map is refused as `owner_unknown`.
