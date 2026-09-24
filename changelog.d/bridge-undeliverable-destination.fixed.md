A queued question whose destination the platform permanently refuses no longer
stays in the retry queue failing the same post once per drain pass. Google
Chat's "This Chat app is not a member of this space" (the app was removed from
the space) and a 404 for a space that no longer exists now raise the new
`UndeliverableError`, and the bridge settles the entry terminal with the
refusal recorded in `give_up_reason` instead of retrying it for a week.
Transient failures keep today's behaviour and stay queued.
