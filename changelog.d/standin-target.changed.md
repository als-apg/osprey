The live stand-in is its own control target, `standin`, beside `live` and
`va`, so `live` always means the facility's own gateways and the build no
longer rewrites them. A deployment starts every session on the stand-in with
`control_system.type: live_standin`; going live is `control_target_set live`.
