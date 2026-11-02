When git is slow, missing or fails inside a repository, the build now says it
could not check whether `.env.variant` is tracked, with git's reason, instead of
saying nothing. When git answers, or no repository encloses the deployment, nothing
changes.
