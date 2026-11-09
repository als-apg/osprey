An optimizer config's `model_file` must lie under the results directory (a run's
saved model or an upload); any other path is refused `422` when the draft is
saved and again at start, so a launch-token holder cannot have the service
unpickle an arbitrary file.
