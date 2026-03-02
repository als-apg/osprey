"""Container Management — re-export facade.

All functions are available from their focused submodules:
- compose_generator: template rendering, build dir setup, compose file creation
- container_lifecycle: deploy up/down/restart/rebuild
- status_display: show_status with Rich output
"""

from osprey.deployment.compose_generator import (  # noqa: F401
    COMPOSE_FILE_NAME,
    OUT_SRC_DIR,
    SERVICES_DIR,
    SRC_DIR,
    TEMPLATE_FILENAME,
    clean_deployment,
    find_existing_compose_files,
    find_service_config,
    get_templates,
    prepare_compose_files,
    render_kernel_templates,
    render_template,
    setup_build_dir,
)
from osprey.deployment.container_lifecycle import (  # noqa: F401
    deploy_down,
    deploy_restart,
    deploy_up,
    rebuild_deployment,
)
from osprey.deployment.status_display import show_status  # noqa: F401
