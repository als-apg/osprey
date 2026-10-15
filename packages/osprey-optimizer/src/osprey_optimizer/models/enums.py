from enum import StrEnum


class JobType(StrEnum):
    """Type of background job tracked by the API."""

    OPTIMIZATION = "optimization"


class JobStatus(StrEnum):
    """Status of a background job."""

    CREATED = "created"
    IDLE = "idle"
    SUBMITTED = "submitted"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    ERROR = "error"
    CANCELLED = "cancelled"

    @property
    def is_terminal(self) -> bool:
        """True for statuses a job can never leave (no worker will change them)."""
        return self in (JobStatus.COMPLETED, JobStatus.ERROR, JobStatus.CANCELLED)


class AlgorithmType(StrEnum):
    """Available optimization algorithms."""

    # Sampling
    RANDOM = "random"
    LATIN_HYPERCUBE = "latin_hypercube"

    # Bayesian optimization
    UPPER_CONFIDENCE_BOUND = "upper_confidence_bound"
    EXPECTED_IMPROVEMENT = "expected_improvement"
    BAXUS = "baxus"

    # Sequential / direct-search
    NELDER_MEAD = "neldermead"
    EXTREMUM_SEEKING = "extremum_seeking"
    RCDS = "rcds"

    # Evolutionary
    NSGA2 = "nsga2"
    CNSGA = "cnsga"


class ConstraintType(StrEnum):
    """Outcome-constraint relation applied to an observable during optimization.

    The values match xopt's VOCS constraint spelling (``[type, threshold]``):
    ``GREATER_THAN`` keeps rows whose observable exceeds the threshold,
    ``LESS_THAN`` keeps rows below it. Both are strict inequalities.
    """

    GREATER_THAN = "GREATER_THAN"
    LESS_THAN = "LESS_THAN"


# These generators reject outcome constraints outright (xopt VOCSError deep in
# construction); shared with the web UI so it can block them before submit.
CONSTRAINT_UNSUPPORTED_ALGORITHMS: frozenset[AlgorithmType] = frozenset(
    {
        AlgorithmType.NELDER_MEAD,
        AlgorithmType.EXTREMUM_SEEKING,
        AlgorithmType.RCDS,
        AlgorithmType.BAXUS,
    }
)

# Algorithms that accept constraints but can misbehave under them (the acquisition
# can go negative); the UI surfaces a non-blocking hint recommending an alternative.
CONSTRAINT_WARN_ALGORITHMS: frozenset[AlgorithmType] = frozenset(
    {AlgorithmType.UPPER_CONFIDENCE_BOUND}
)


class SurrogateModelType(StrEnum):
    """Available surrogate model types for Bayesian optimization."""

    DEFAULT = "default"
    RANDOM_FOREST = "random_forest"
    EXTRA_TREES = "extra_trees"
    GAUSSIAN_PROCESS = "gaussian_process"
    XOPT_GP = "xopt_gp"


class EnvironmentType(StrEnum):
    """Available environment types."""

    PRODUCTION = "production"
    SIMULATION = "simulation"
    TESTING = "testing"
    READONLY = "readonly"
