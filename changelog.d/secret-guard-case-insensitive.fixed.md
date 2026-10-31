Executed code can no longer open a secret `.env` file by spelling it in another
case. On a case-insensitive filesystem (macOS's default) `.ENV`, `.Env.shared`
or a recased project directory opened the secret file, because the filesystem
guard compared the name and the secret roots case-sensitively; both are now
compared casefolded.
