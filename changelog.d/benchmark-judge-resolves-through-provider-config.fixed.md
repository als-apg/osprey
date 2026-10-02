The channel-finder benchmark's coverage judge runs on a provider the project configures under
`api.providers` (the benchmarked model's own, or another named with `judge_provider=`) instead of whichever
vendor key happens to be exported. A judge the project cannot run is refused when `BenchmarkRunner` is
built, and a judge call that fails fails its query instead of being rescored by substring match.
