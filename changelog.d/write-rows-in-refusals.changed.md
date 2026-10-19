A `channel_write` call where no value reached any channel now carries its rows under
`details.results`. A batch in which any write was sent and not taken is refused as
`nothing_answered` instead of `internal_error`.
