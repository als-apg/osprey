Naming your own image in `services.virtual_accelerator.image` (or
`services.live_standin.image`) now runs that image as named. The service renders
without a `build:` block, so a deploy no longer rebuilds OSPREY's recipe and
tags it with your image's name.
