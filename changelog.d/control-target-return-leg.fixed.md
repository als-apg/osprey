A session can always come home. Returning to the deployment's own baseline
no longer fails the eligibility checks when that target's connector block
carries no gateways, no selectable gateway role or no probe channel — the shape
a block reached over Channel Access has, and one a baseline on another
protocol, or a block still half authored, may never fill in. A deployment
baselined on the live machine whose block leaves ``probe_channel`` commented
out could otherwise rehearse on the simulator once and stay there until the
controls server restarted. Coming home still requires the target's connector
block, still refuses a stand-in block pointed anywhere but this deployment's
own stand-in, and is proven with the baseline's probe channel wherever the
block names one. A baseline whose block derives no endpoint at all is still
refused as not probed while a controls server is live.
