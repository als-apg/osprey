"""Factory for creating code generators based on configuration and registry.

This module provides the factory pattern implementation for code generator selection,
integrated with Osprey's registry system. It enables configuration-driven selection
of code generation strategies, supporting framework built-ins and application-registered
custom generators with automatic fallback on missing dependencies.

The factory leverages Osprey's registry to discover generators registered by:
- Framework (built-in generators like legacy and claude_code)
- Applications (custom domain-specific generators)

This provides several key benefits:
- Consistent with framework's component registration patterns
- Automatic discovery of application generators
- Graceful fallback when optional dependencies aren't available
- Centralized generator creation logic
- Clear separation between framework and application generators

Configuration Example::

    osprey:
      execution:
        code_generator: "basic"  # or "claude_code", or custom name
        generators:
          basic:
            model_config_name: "python_code_generator"
          claude_code:
            profile: "balanced"

Application Registry Example::

    # In applications/myapp/registry.py
    RegistryConfig(
        code_generators=[
            CodeGeneratorRegistration(
                name="domain_specific",
                module_path="applications.myapp.generators.domain",
                class_name="DomainGenerator",
                description="Domain-specific code generator"
            )
        ]
    )

.. note::
   Generators are discovered through the registry system automatically.
   Applications register custom generators via their registry.py module.

.. seealso::
   :class:`osprey.registry.base.CodeGeneratorRegistration`
   :class:`osprey.services.python_executor.generation.interface.CodeGenerator`
   :class:`osprey.services.python_executor.generation.basic_generator.BasicLLMCodeGenerator`
   :class:`osprey.services.python_executor.generation.claude_code_generator.ClaudeCodeGenerator`

Examples:
    Using default configuration::

        >>> generator = create_code_generator()
        >>> # Returns configured generator from registry

    Using custom configuration::

        >>> config = {
        ...     "execution": {
        ...         "code_generator": "claude_code",
        ...         "generators": {
            ...             "claude_code": {"profile": "fast"}
        ...         }
        ...     }
        ... }
        >>> generator = create_code_generator(config)

    Registering custom generator via registry::

        >>> # In your application's registry.py
        >>> RegistryConfig(
        ...     code_generators=[
        ...         CodeGeneratorRegistration(
        ...             name="my_gen",
        ...             module_path="myapp.generators",
        ...             class_name="MyGenerator",
        ...             description="Custom generator"
        ...         )
        ...     ]
        ... )
        >>> # Then in config: code_generator: "my_gen"
"""

from typing import Any

from osprey.utils.config import get_full_configuration
from osprey.utils.logger import get_logger

logger = get_logger("generator_factory")


def create_code_generator(config: dict[str, Any] | None = None):
    """Create code generator based on configuration and registry.

    Creates and returns a code generator instance by discovering registered
    generators through the Osprey registry system. Supports framework generators
    (basic, claude_code) and application-registered custom generators.
    Automatically falls back to basic generator if the requested generator
    is unavailable due to missing optional dependencies.

    The factory discovers generators registered via:
    - Framework registry (built-in generators)
    - Application registries (custom generators)

    Args:
        config: Configuration dictionary. If None, uses global configuration
               from get_full_configuration().

    Returns:
        CodeGenerator instance (discovered from registry)

    Raises:
        ValueError: If specified generator type is not found in registry
        ImportError: If generator's module cannot be imported (non-optional deps)

    Configuration Structure::

        osprey:
          execution:
            code_generator: "basic"  # Generator name from registry
            generators:
              basic:
                model_config_name: "python_code_generator"
                # or inline: provider, model_id, etc.
              claude_code:
                profile: "fast"  # fast (DEFAULT, single-phase) | robust (multi-phase)

    .. note::
       The function gracefully handles generator initialization failures.
       If a generator cannot be initialized, it logs a warning and falls back
       to the basic generator.

    .. note::
       Generators are discovered through the registry system automatically.
       Applications can register custom generators via their registry.py module.

    .. seealso::
       :class:`osprey.registry.base.CodeGeneratorRegistration`
       :class:`osprey.services.python_executor.generation.interface.CodeGenerator`

    Examples:
        Using default configuration::

            >>> generator = create_code_generator()
            >>> code = await generator.generate_code(request, [])

        Using custom configuration::

            >>> config = {"execution": {"code_generator": "claude_code"}}
            >>> generator = create_code_generator(config)

        Application custom generator (via registry)::

            >>> # In applications/myapp/registry.py:
            >>> RegistryConfig(
            ...     code_generators=[
            ...         CodeGeneratorRegistration(
            ...             name="domain_gen",
            ...             module_path="applications.myapp.generators",
            ...             class_name="DomainGenerator",
            ...             description="Custom generator"
            ...         )
            ...     ]
            ... )
            >>> # Then in config: code_generator: "domain_gen"
            >>> generator = create_code_generator()  # Finds domain_gen
    """
    if config is None:
        config = get_full_configuration()

    # Extract generator configuration
    execution_config = config.get("execution", {})
    generator_type = execution_config.get("code_generator", "basic")
    generators_config = execution_config.get("generators", {})
    generator_config = generators_config.get(generator_type, {})

    logger.info(f"Creating code generator: {generator_type}")

    # Try registry-based discovery first
    try:
        from osprey.registry import get_registry
        registry = get_registry()

        # Get registered code generators from registry
        # Registry stores them in _registries['code_generators'] during initialization
        generator_registrations = registry._registries.get('code_generators', {})

        if generator_type in generator_registrations:
            registry_entry = generator_registrations[generator_type]
            registration = registry_entry['registration']
            generator_class = registry_entry['class']

            logger.info(f"Found generator '{generator_type}' in registry")
            logger.info(f"Using {registration.description}")

            # Instantiate and return
            return generator_class(model_config=generator_config)

        # Generator not found in registry
        else:
            available = list(generator_registrations.keys())
            logger.warning(
                f"Generator '{generator_type}' not found in registry. "
                f"Available: {available}. Trying direct import fallback..."
            )

    except Exception as e:
        # Registry not available or other error - fall through to direct import
        logger.warning(f"Registry lookup failed: {e}. Trying direct import fallback.")

    # Fallback: Direct import for backwards compatibility and robustness
    if generator_type == "claude_code":
        try:
            from .claude_code_generator import ClaudeCodeGenerator
            logger.info("Using Claude Code generator (direct import)")
            return ClaudeCodeGenerator(model_config=generator_config)
        except ImportError as e:
            logger.warning(f"Claude Code SDK not available: {e}")
            logger.info("Falling back to basic generator")
            generator_type = "basic"
            generator_config = generators_config.get("basic", {})

    # Mock generator (for testing)
    if generator_type == "mock":
        from .mock_generator import MockCodeGenerator
        logger.info("Using mock generator (for testing)")
        return MockCodeGenerator(model_config=generator_config)

    # Basic generator (always available) - also support legacy name for backwards compatibility
    if generator_type in ("basic", "legacy"):
        from .basic_generator import BasicLLMCodeGenerator
        if generator_type == "legacy":
            logger.warning("Generator name 'legacy' is deprecated, use 'basic' instead")
        logger.info("Using basic LLM generator")
        return BasicLLMCodeGenerator(model_config=generator_config)

    # Unknown generator type
    raise ValueError(
        f"Generator '{generator_type}' not available. "
        f"Available built-in generators: 'basic', 'claude_code', 'mock'. "
        f"Register custom generators via your application's registry.py"
    )
