"""
Cross-validation tests for the v0.11 migration document.

Validates that every search_pattern in v0.11.yml matches real code patterns
from both vanilla (template-generated) and ALS-style (exotic patterns) v0.10 projects,
and that facility-specific files are NOT incorrectly matched by removal patterns.
"""

import re
from pathlib import Path

import pytest
import yaml

# ---------------------------------------------------------------------------
# Load migration document
# ---------------------------------------------------------------------------

MIGRATION_DOC_PATH = (
    Path(__file__).parent.parent.parent
    / "src"
    / "osprey"
    / "assist"
    / "tasks"
    / "migrate"
    / "versions"
    / "v0.11.yml"
)


@pytest.fixture(scope="module")
def migration_doc():
    """Load and parse the v0.11 migration document."""
    with open(MIGRATION_DOC_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Schema definition (from schema.yml)
# ---------------------------------------------------------------------------

CHANGE_TYPE_REQUIRED = {
    "method_rename": {"old", "new", "search_pattern", "risk", "automatable"},
    "signature_change": {
        "method",
        "old_signature",
        "new_signature",
        "search_pattern",
        "risk",
        "automatable",
    },
    "import_change": {"old_import", "new_import", "search_pattern", "risk", "automatable"},
    "class_rename": {"old", "new", "search_pattern", "risk", "automatable"},
    "config_change": {"description", "file_pattern", "risk", "automatable"},
    "behavior_change": {"description", "risk", "automatable"},
    "removal": {"removed", "risk", "automatable"},
    "addition": {"description"},
}

VALID_RISK_LEVELS = {"high", "medium", "low"}
VALID_AUTOMATABLE = {True, False, "partial"}

# Required top-level fields
REQUIRED_TOP_LEVEL = {"version", "release_date", "summary", "migrates_from", "changes"}


# ---------------------------------------------------------------------------
# Vanilla v0.10 project fixture (template-generated)
# ---------------------------------------------------------------------------

VANILLA_FILES = {
    # capabilities/ — template-generated with relative imports
    "src/test_facility/capabilities/__init__.py": """\
from test_facility.capabilities.channel_finding import ChannelFindingCapability
from test_facility.capabilities.channel_read import ChannelReadCapability
from test_facility.capabilities.channel_write import ChannelWriteCapability
from test_facility.capabilities.archiver_retrieval import ArchiverRetrievalCapability
""",
    "src/test_facility/capabilities/channel_finding.py": '''\
"""Channel Finding Capability"""
from __future__ import annotations
from osprey.base.capability import BaseCapability
from osprey.base.decorators import capability_node
from ..context_classes import ChannelAddressesContext
from ..services.channel_finder.service import ChannelFinderService

class ChannelFindingCapability(BaseCapability):
    name = "channel_finding"
''',
    "src/test_facility/capabilities/channel_read.py": '''\
"""Channel Read Capability"""
from osprey.base.capability import BaseCapability
from ..context_classes import ChannelValue, ChannelValuesContext

class ChannelReadCapability(BaseCapability):
    name = "channel_read"
''',
    "src/test_facility/capabilities/channel_write.py": '''\
"""Channel Write Capability"""
from osprey.base.capability import BaseCapability
from ..context_classes import ChannelWriteResult, ChannelWriteResultsContext, WriteVerificationInfo

class ChannelWriteCapability(BaseCapability):
    name = "channel_write"
''',
    "src/test_facility/capabilities/archiver_retrieval.py": '''\
"""Archiver Retrieval Capability"""
from osprey.base.capability import BaseCapability
from ..context_classes import ArchiverDataContext

class ArchiverRetrievalCapability(BaseCapability):
    name = "archiver_retrieval"
''',
    # tests/ — test files use absolute imports to capabilities and context classes
    "tests/test_capabilities.py": '''\
"""Test capabilities."""
from test_facility.capabilities.channel_finding import ChannelFindingCapability
from test_facility.capabilities.channel_read import ChannelReadCapability
from test_facility.capabilities.channel_write import ChannelWriteCapability
from test_facility.capabilities.archiver_retrieval import ArchiverRetrievalCapability
from test_facility.context_classes import ChannelAddressesContext
from test_facility.context_classes import ChannelValuesContext
from test_facility.context_classes import ChannelWriteResultsContext
from test_facility.context_classes import ArchiverDataContext
from test_facility.context_classes import ChannelAddressesContext, ChannelValuesContext, ChannelWriteResultsContext
''',
    # context_classes.py
    "src/test_facility/context_classes.py": '''\
"""Context classes for Test Facility."""
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, ClassVar
from datetime import datetime
from osprey.context import CapabilityContext

class ChannelAddressesContext(CapabilityContext):
    CONTEXT_TYPE: ClassVar[str] = "CHANNEL_ADDRESSES"
    channels: List[str]
    original_query: str

class ChannelValue(BaseModel):
    value: str
    timestamp: datetime
    units: str

class ChannelValuesContext(CapabilityContext):
    CONTEXT_TYPE: ClassVar[str] = "CHANNEL_VALUES"
    channel_values: Dict[str, ChannelValue]

class WriteVerificationInfo(BaseModel):
    level: str
    verified: bool

class ChannelWriteResult(BaseModel):
    channel_address: str
    value_written: Any
    success: bool

class ChannelWriteResultsContext(CapabilityContext):
    CONTEXT_TYPE: ClassVar[str] = "CHANNEL_WRITE_RESULTS"
    results: List[ChannelWriteResult]
    total_writes: int
    successful_count: int
    failed_count: int

class ArchiverDataContext(CapabilityContext):
    CONTEXT_TYPE: ClassVar[str] = "ARCHIVER_DATA"
    timestamps: List[datetime]
    precision_ms: int
    time_series_data: Dict[str, List[float]]
    available_channels: List[str]
''',
    # registry.py — v0.10 style with CapabilityRegistration + ContextClassRegistration
    "src/test_facility/registry.py": '''\
"""Component registry for Test Facility."""
from osprey.registry import RegistryConfigProvider, extend_framework_registry, CapabilityRegistration, ContextClassRegistration, FrameworkPromptProviderRegistration, RegistryConfig

class TestFacilityRegistryProvider(RegistryConfigProvider):
    def get_registry_config(self) -> RegistryConfig:
        return extend_framework_registry(
            capabilities=[
                CapabilityRegistration(
                    name="channel_finding",
                    module_path="test_facility.capabilities.channel_finding",
                    class_name="ChannelFindingCapability",
                    provides=["CHANNEL_ADDRESSES"],
                    requires=[]
                ),
                CapabilityRegistration(
                    name="channel_read",
                    module_path="test_facility.capabilities.channel_read",
                    class_name="ChannelReadCapability",
                    provides=["CHANNEL_VALUES"],
                    requires=["CHANNEL_ADDRESSES"]
                ),
                CapabilityRegistration(
                    name="channel_write",
                    module_path="test_facility.capabilities.channel_write",
                    class_name="ChannelWriteCapability",
                    provides=["CHANNEL_WRITE_RESULTS"],
                    requires=["CHANNEL_ADDRESSES"]
                ),
                CapabilityRegistration(
                    name="archiver_retrieval",
                    module_path="test_facility.capabilities.archiver_retrieval",
                    class_name="ArchiverRetrievalCapability",
                    provides=["ARCHIVER_DATA"],
                    requires=["CHANNEL_ADDRESSES", "TIME_RANGE"]
                ),
            ],
            context_classes=[
                ContextClassRegistration(
                    context_type="CHANNEL_ADDRESSES",
                    module_path="test_facility.context_classes",
                    class_name="ChannelAddressesContext"
                ),
                ContextClassRegistration(
                    context_type="CHANNEL_VALUES",
                    module_path="test_facility.context_classes",
                    class_name="ChannelValuesContext"
                ),
                ContextClassRegistration(
                    context_type="CHANNEL_WRITE_RESULTS",
                    module_path="test_facility.context_classes",
                    class_name="ChannelWriteResultsContext"
                ),
                ContextClassRegistration(
                    context_type="ARCHIVER_DATA",
                    module_path="test_facility.context_classes",
                    class_name="ArchiverDataContext"
                ),
            ],
            framework_prompt_providers=[
                FrameworkPromptProviderRegistration(
                    module_path="test_facility.framework_prompts",
                    prompt_builders={
                        "python": "ControlSystemPythonPromptBuilder",
                        "task_extraction": "ControlSystemTaskExtractionPromptBuilder"
                    }
                )
            ]
        )
''',
    # services/channel_finder/ — representative files
    "src/test_facility/services/__init__.py": "",
    "src/test_facility/services/channel_finder/__init__.py": "",
    "src/test_facility/services/channel_finder/service.py": '''\
"""Channel Finder Service."""
class ChannelFinderService:
    pass
''',
    "src/test_facility/services/channel_finder/core/__init__.py": "",
    "src/test_facility/services/channel_finder/core/models.py": "# models",
    "src/test_facility/services/channel_finder/pipelines/__init__.py": "",
    "src/test_facility/services/channel_finder/utils/config.py": "# config",
    "src/test_facility/services/channel_finder/utils/prompt_loader.py": "# prompt_loader",
    # data/tools/ — with template-style absolute imports
    "src/test_facility/data/tools/__init__.py": "",
    "src/test_facility/data/tools/build_channel_database.py": '''\
"""Build channel database."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from test_facility.services.channel_finder.utils.config import get_config, resolve_path
''',
    "src/test_facility/data/tools/validate_database.py": '''\
"""Validate database."""
from test_facility.services.channel_finder.utils.config import get_config, resolve_path
from test_facility.services.channel_finder.databases import TemplateChannelDatabase
''',
    "src/test_facility/data/tools/preview_database.py": '''\
"""Preview database."""
from test_facility.services.channel_finder.utils.config import get_config, resolve_path
''',
    "src/test_facility/data/tools/llm_channel_namer.py": '''\
"""LLM channel namer."""
from test_facility.services.channel_finder.utils.config import get_config, load_config
''',
    # framework_prompts/
    "src/test_facility/framework_prompts/__init__.py": '''\
"""Framework prompt customizations."""
from .python import ControlSystemPythonPromptBuilder
from .task_extraction import ControlSystemTaskExtractionPromptBuilder
''',
    # config.yml
    "config.yml": """\
llm:
  default_provider: anthropic
channel_finder:
  pipeline: hierarchical
  prompts:
    path: "prompts/hierarchical"
prompt_builders:
  python: ControlSystemPythonPromptBuilder
""",
}


# ---------------------------------------------------------------------------
# ALS-style v0.10 project fixture (exotic patterns)
# ---------------------------------------------------------------------------

ALS_STYLE_FILES = {
    # capabilities/ — with relative imports (from ..)
    # Real ALS: empty __init__.py with only docstring (no re-exports)
    "src/control_assistant/capabilities/__init__.py": '''\
"""
Capabilities for control assistant.

Thin wrappers around service layer that handle Osprey framework integration.
"""
''',
    "src/control_assistant/capabilities/channel_finding.py": '''\
"""Channel Finding Capability"""
from __future__ import annotations
from osprey.base.capability import BaseCapability
from ..context_classes import ChannelAddressesContext
from ..services.channel_finder.service import ChannelFinderService

class ChannelFindingCapability(BaseCapability):
    name = "channel_finding"
''',
    "src/control_assistant/capabilities/channel_read.py": '''\
"""Channel Read Capability"""
from osprey.base.capability import BaseCapability
from ..context_classes import ChannelValue, ChannelValuesContext

class ChannelReadCapability(BaseCapability):
    name = "channel_read"
''',
    # channel_write.py — exotic patterns from real ALS:
    #   conditional try/except import, deferred imports inside method body
    "src/control_assistant/capabilities/channel_write.py": '''\
"""Channel Write Capability"""
from osprey.base.capability import BaseCapability
from osprey.base.decorators import capability_node
from osprey.connectors.factory import ConnectorFactory
from ..context_classes import ChannelWriteResult, ChannelWriteResultsContext, WriteVerificationInfo

# Conditional try/except import (real ALS pattern)
try:
    from osprey.models import get_chat_completion
except ImportError:
    get_chat_completion = None

class ChannelWriteCapability(BaseCapability):
    name = "channel_write"

    def _get_write_parsing_system_prompt(self, task_objective):
        # Deferred import inside method body (real ALS pattern)
        from osprey.registry import get_registry
        registry = get_registry()
        return f"Parse write operations for {task_objective}"

    async def execute(self, plan):
        # Another deferred import (real ALS pattern)
        from osprey.utils.config import get_config_value
        writes_enabled = get_config_value("control_system.writes_enabled", False)
        return writes_enabled
''',
    "src/control_assistant/capabilities/archiver_retrieval.py": '''\
"""Archiver Retrieval Capability"""
from osprey.base.capability import BaseCapability
from ..context_classes import ArchiverDataContext

class ArchiverRetrievalCapability(BaseCapability):
    name = "archiver_retrieval"
''',
    # Extra facility-specific capability — must NOT be touched
    "src/control_assistant/capabilities/live_monitoring.py": '''\
"""Live Monitoring Capability"""
from osprey.base.capability import BaseCapability
from control_assistant.services.launcher.service import LauncherService
from control_assistant.services.launcher.models import LauncherServiceRequest

class LiveMonitoringCapability(BaseCapability):
    name = "live_monitoring"
''',
    # context_classes.py — includes exotic patterns from real ALS:
    #   CONTEXT_CATEGORY ClassVar, get_summary()/get_access_details() methods,
    #   extra fields on WriteVerificationInfo/ChannelWriteResult,
    #   channel_count property, facility-specific LauncherResultsContext
    "src/control_assistant/context_classes.py": '''\
"""Context classes for ALS Assistant v2."""
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, ClassVar
from datetime import datetime
from osprey.context import CapabilityContext

class ChannelAddressesContext(CapabilityContext):
    CONTEXT_TYPE: ClassVar[str] = "CHANNEL_ADDRESSES"
    CONTEXT_CATEGORY: ClassVar[str] = "METADATA"
    channels: List[str]
    original_query: str

    def get_access_details(self, key: str) -> Dict[str, Any]:
        return {"channels": self.channels, "total_available": len(self.channels)}

    def get_summary(self) -> Dict[str, Any]:
        return {"type": "Channel Addresses", "total_channels": len(self.channels)}

class ChannelValue(BaseModel):
    value: str
    timestamp: datetime
    units: str

class WriteVerificationInfo(BaseModel):
    level: str
    verified: bool
    readback_value: Optional[float] = None
    tolerance_used: Optional[float] = None
    notes: Optional[str] = None

class ChannelWriteResult(BaseModel):
    channel_address: str
    value_written: Any
    success: bool
    verification: Optional[WriteVerificationInfo] = None
    error_message: Optional[str] = None

class ChannelValuesContext(CapabilityContext):
    CONTEXT_TYPE: ClassVar[str] = "CHANNEL_VALUES"
    CONTEXT_CATEGORY: ClassVar[str] = "COMPUTATIONAL_DATA"
    channel_values: Dict[str, ChannelValue]

    @property
    def channel_count(self) -> int:
        return len(self.channel_values)

    def get_summary(self) -> Dict[str, Any]:
        return {"type": "Channel Values", "count": self.channel_count}

class ChannelWriteResultsContext(CapabilityContext):
    CONTEXT_TYPE: ClassVar[str] = "CHANNEL_WRITE_RESULTS"
    CONTEXT_CATEGORY: ClassVar[str] = "OPERATIONAL_DATA"
    results: List[ChannelWriteResult]
    total_writes: int
    successful_count: int
    failed_count: int

class ArchiverDataContext(CapabilityContext):
    CONTEXT_TYPE: ClassVar[str] = "ARCHIVER_DATA"
    CONTEXT_CATEGORY: ClassVar[str] = "COMPUTATIONAL_DATA"
    timestamps: List[datetime]
    precision_ms: int
    time_series_data: Dict[str, List[float]]
    available_channels: List[str]

class LauncherResultsContext(CapabilityContext):
    CONTEXT_TYPE: ClassVar[str] = "LAUNCHER_RESULTS"
    launch_uri: str
    command_name: str
    command_description: str
    success: bool
    channel_count: int
    monitoring_type: str
''',
    # tests/ — absolute imports for capabilities and context classes
    "tests/test_capabilities.py": '''\
"""Test capabilities."""
from control_assistant.capabilities.channel_finding import ChannelFindingCapability
from control_assistant.capabilities.channel_read import ChannelReadCapability
from control_assistant.capabilities.channel_write import ChannelWriteCapability
from control_assistant.capabilities.archiver_retrieval import ArchiverRetrievalCapability
from control_assistant.context_classes import ChannelAddressesContext
from control_assistant.context_classes import ChannelValuesContext
from control_assistant.context_classes import ChannelWriteResultsContext
from control_assistant.context_classes import ArchiverDataContext
from control_assistant.context_classes import ChannelAddressesContext, ChannelValuesContext, ChannelWriteResultsContext
''',
    # registry.py — with description= fields (real ALS pattern),
    #   extra live_monitoring, LAUNCHER_RESULTS, prompt builders
    "src/control_assistant/registry.py": '''\
"""Component registry for ALS Assistant v2."""
from osprey.registry import (
    RegistryConfigProvider,
    extend_framework_registry,
    CapabilityRegistration,
    ContextClassRegistration,
    FrameworkPromptProviderRegistration,
    RegistryConfig
)

class ALSAssistantV2RegistryProvider(RegistryConfigProvider):
    def get_registry_config(self) -> RegistryConfig:
        return extend_framework_registry(
            capabilities=[
                CapabilityRegistration(
                    name="channel_finding",
                    module_path="control_assistant.capabilities.channel_finding",
                    class_name="ChannelFindingCapability",
                    description="Find control system channels using semantic search",
                    provides=["CHANNEL_ADDRESSES"],
                    requires=[]
                ),
                CapabilityRegistration(
                    name="channel_read",
                    module_path="control_assistant.capabilities.channel_read",
                    class_name="ChannelReadCapability",
                    description="Read current values from control system channels",
                    provides=["CHANNEL_VALUES"],
                    requires=["CHANNEL_ADDRESSES"]
                ),
                CapabilityRegistration(
                    name="channel_write",
                    module_path="control_assistant.capabilities.channel_write",
                    class_name="ChannelWriteCapability",
                    description="Write values to control system channels",
                    provides=["CHANNEL_WRITE_RESULTS"],
                    requires=["CHANNEL_ADDRESSES"]
                ),
                CapabilityRegistration(
                    name="archiver_retrieval",
                    module_path="control_assistant.capabilities.archiver_retrieval",
                    class_name="ArchiverRetrievalCapability",
                    description="Query historical time-series data from the archiver",
                    provides=["ARCHIVER_DATA"],
                    requires=["CHANNEL_ADDRESSES", "TIME_RANGE"]
                ),
                CapabilityRegistration(
                    name="live_monitoring",
                    module_path="control_assistant.capabilities.live_monitoring",
                    class_name="LiveMonitoringCapability",
                    description="Launch Phoebus Data Browser for real-time monitoring",
                    provides=[],
                    requires=["CHANNEL_ADDRESSES"]
                ),
            ],
            context_classes=[
                ContextClassRegistration(
                    context_type="CHANNEL_ADDRESSES",
                    module_path="control_assistant.context_classes",
                    class_name="ChannelAddressesContext"
                ),
                ContextClassRegistration(
                    context_type="CHANNEL_VALUES",
                    module_path="control_assistant.context_classes",
                    class_name="ChannelValuesContext"
                ),
                ContextClassRegistration(
                    context_type="CHANNEL_WRITE_RESULTS",
                    module_path="control_assistant.context_classes",
                    class_name="ChannelWriteResultsContext"
                ),
                ContextClassRegistration(
                    context_type="ARCHIVER_DATA",
                    module_path="control_assistant.context_classes",
                    class_name="ArchiverDataContext"
                ),
                ContextClassRegistration(
                    context_type="LAUNCHER_RESULTS",
                    module_path="control_assistant.context_classes",
                    class_name="LauncherResultsContext"
                ),
            ],
            framework_prompt_providers=[
                FrameworkPromptProviderRegistration(
                    module_path="control_assistant.framework_prompts",
                    prompt_builders={
                        "python": "ControlSystemPythonPromptBuilder",
                        "task_extraction": "ControlSystemTaskExtractionPromptBuilder"
                    }
                )
            ]
        )
''',
    # services/channel_finder/ — representative files
    "src/control_assistant/services/__init__.py": "",
    "src/control_assistant/services/channel_finder/__init__.py": "",
    "src/control_assistant/services/channel_finder/service.py": '''\
"""Channel Finder Service."""
class ChannelFinderService:
    pass
''',
    "src/control_assistant/services/channel_finder/core/__init__.py": "",
    "src/control_assistant/services/channel_finder/core/models.py": "# models",
    "src/control_assistant/services/channel_finder/pipelines/__init__.py": "",
    # Triple-dot relative imports (real ALS pattern: pipelines/<name>/ → ...core)
    "src/control_assistant/services/channel_finder/pipelines/hierarchical/__init__.py": "",
    "src/control_assistant/services/channel_finder/pipelines/hierarchical/pipeline.py": '''\
"""Hierarchical pipeline."""
from ...core.base_pipeline import BasePipeline
from ...core.models import ChannelFinderResult, ChannelInfo, QuerySplitterOutput
from ...utils.prompt_loader import load_prompts
class HierarchicalPipeline(BasePipeline):
    pass
''',
    "src/control_assistant/services/channel_finder/pipelines/in_context/__init__.py": "",
    "src/control_assistant/services/channel_finder/pipelines/in_context/pipeline.py": '''\
"""In-context pipeline."""
from ...core.base_pipeline import BasePipeline
from ...core.models import ChannelFinderResult
from ...utils.prompt_loader import load_prompts
class InContextPipeline(BasePipeline):
    pass
''',
    "src/control_assistant/services/channel_finder/utils/config.py": "# config",
    "src/control_assistant/services/channel_finder/utils/prompt_loader.py": "# prompt_loader",
    # examples/ directory (real ALS has custom_database_example.py, custom_pipeline_example.py)
    "src/control_assistant/services/channel_finder/examples/__init__.py": "",
    "src/control_assistant/services/channel_finder/examples/custom_database_example.py": '''\
"""Example custom database implementation."""
from ..core.models import ChannelInfo
''',
    "src/control_assistant/services/channel_finder/examples/custom_pipeline_example.py": '''\
"""Example custom pipeline implementation."""
from ...core.base_pipeline import BasePipeline
''',
    "src/control_assistant/services/channel_finder/benchmarks/__init__.py": "",
    "src/control_assistant/services/channel_finder/benchmarks/runner.py": '''\
"""Benchmark runner."""
from control_assistant.services.channel_finder.service import ChannelFinderService
''',
    # benchmarks/cli.py — with sys.path.insert hack (real ALS pattern)
    "src/control_assistant/services/channel_finder/benchmarks/cli.py": '''\
"""Benchmark CLI."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))
from control_assistant.services.channel_finder.benchmarks.runner import BenchmarkRunner
from control_assistant.services.channel_finder.utils.config import get_config
''',
    "src/control_assistant/services/channel_finder/cli.py": '''\
"""Channel finder CLI."""
from control_assistant.services.channel_finder.service import ChannelFinderService
''',
    # Extra facility-specific service — must NOT be touched
    "src/control_assistant/services/launcher/__init__.py": "",
    "src/control_assistant/services/launcher/service.py": '''\
"""Launcher Service for Phoebus Data Browser."""
class LauncherService:
    pass
''',
    "src/control_assistant/services/launcher/models.py": '''\
"""Launcher models."""
class LauncherServiceRequest:
    pass
''',
    "src/control_assistant/services/launcher/utils.py": '''\
"""Launcher utilities."""
''',
    # data/tools/ — with sys.path.insert hacks and absolute imports (real ALS pattern)
    "src/control_assistant/data/tools/__init__.py": "",
    "src/control_assistant/data/tools/build_channel_database.py": '''\
"""Build channel database."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from control_assistant.services.channel_finder.utils.config import get_config, resolve_path
''',
    "src/control_assistant/data/tools/validate_database.py": '''\
"""Validate database."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from control_assistant.services.channel_finder.utils.config import get_config, resolve_path
from control_assistant.services.channel_finder.databases import TemplateChannelDatabase
''',
    "src/control_assistant/data/tools/preview_database.py": '''\
"""Preview database."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from control_assistant.services.channel_finder.utils.config import get_config, resolve_path
from control_assistant.services.channel_finder.databases import TemplateChannelDatabase, HierarchicalChannelDatabase, MiddleLayerDatabase
''',
    "src/control_assistant/data/tools/llm_channel_namer.py": '''\
"""LLM channel namer."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from control_assistant.services.channel_finder.utils.config import get_config, load_config
''',
    # framework_prompts/ — no channel_finder entries
    "src/control_assistant/framework_prompts/__init__.py": '''\
"""Framework prompt customizations."""
from .python import ControlSystemPythonPromptBuilder
from .task_extraction import ControlSystemTaskExtractionPromptBuilder
''',
    # config.yml
    "config.yml": """\
llm:
  default_provider: anthropic
channel_finder:
  pipeline: hierarchical
  prompts:
    path: "prompts/hierarchical"
prompt_builders:
  python: ControlSystemPythonPromptBuilder
""",
}


# ---------------------------------------------------------------------------
# v0.11 project fixture (post-migration) — patterns should NOT match here
# ---------------------------------------------------------------------------

V011_FILES = {
    # No capabilities/ directory (all native now)
    # No context_classes.py
    # No services/channel_finder/
    # No data/tools/
    # registry.py — v0.11 style
    "src/test_facility/registry.py": '''\
"""Component registry for Test Facility."""
from osprey.registry import (
    RegistryConfigProvider,
    extend_framework_registry,
    FrameworkPromptProviderRegistration,
    RegistryConfig
)

class TestFacilityRegistryProvider(RegistryConfigProvider):
    def get_registry_config(self) -> RegistryConfig:
        return extend_framework_registry(
            framework_prompt_providers=[
                FrameworkPromptProviderRegistration(
                    module_path="test_facility.framework_prompts",
                    prompt_builders={
                        "python": "ControlSystemPythonPromptBuilder",
                        "task_extraction": "ControlSystemTaskExtractionPromptBuilder",
                        "channel_finder_in_context": "FacilityInContextPromptBuilder",
                        "channel_finder_hierarchical": "FacilityHierarchicalPromptBuilder",
                        "channel_finder_middle_layer": "FacilityMiddleLayerPromptBuilder",
                    }
                )
            ]
        )
''',
    "src/test_facility/framework_prompts/__init__.py": '''\
"""Framework prompt customizations."""
from .python import ControlSystemPythonPromptBuilder
from .task_extraction import ControlSystemTaskExtractionPromptBuilder
from .channel_finder.in_context import FacilityInContextPromptBuilder
from .channel_finder.hierarchical import FacilityHierarchicalPromptBuilder
from .channel_finder.middle_layer import FacilityMiddleLayerPromptBuilder
''',
    "config.yml": """\
llm:
  default_provider: anthropic
channel_finder:
  pipeline: hierarchical
""",
}


# ---------------------------------------------------------------------------
# Facility-specific files that must NOT be matched by removal patterns
# ---------------------------------------------------------------------------

FACILITY_SPECIFIC_FILES = {
    "src/control_assistant/capabilities/live_monitoring.py": ALS_STYLE_FILES[
        "src/control_assistant/capabilities/live_monitoring.py"
    ],
    "src/control_assistant/services/launcher/__init__.py": "",
    "src/control_assistant/services/launcher/service.py": ALS_STYLE_FILES[
        "src/control_assistant/services/launcher/service.py"
    ],
    "src/control_assistant/services/launcher/models.py": ALS_STYLE_FILES[
        "src/control_assistant/services/launcher/models.py"
    ],
    "src/control_assistant/services/launcher/utils.py": ALS_STYLE_FILES[
        "src/control_assistant/services/launcher/utils.py"
    ],
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _get_changes_with_search_pattern(doc: dict) -> list[dict]:
    """Get all changes that have a search_pattern field."""
    return [c for c in doc["changes"] if "search_pattern" in c]


def _get_changes_by_type(doc: dict, change_type: str) -> list[dict]:
    """Get all changes of a specific type."""
    return [c for c in doc["changes"] if c["type"] == change_type]


def _match_pattern_against_files(pattern: str, files: dict[str, str]) -> list[tuple[str, str]]:
    """Return list of (path, matched_line) for all matches."""
    compiled = re.compile(pattern)
    matches = []
    for path, content in files.items():
        for line in content.splitlines():
            if compiled.search(line):
                matches.append((path, line))
    return matches


def _match_pattern_against_paths(pattern: str, files: dict[str, str]) -> list[str]:
    """Return list of file paths that match the pattern in their path."""
    compiled = re.compile(pattern)
    return [path for path in files if compiled.search(path)]


# ===========================================================================
# Test 1: Search patterns match vanilla v0.10 project
# ===========================================================================


class TestSearchPatternsMatchVanillaV010:
    """Every search_pattern must match at least one file in the vanilla project."""

    def test_all_search_patterns_have_matches(self, migration_doc):
        changes_with_patterns = _get_changes_with_search_pattern(migration_doc)
        assert len(changes_with_patterns) > 0, "No changes with search_pattern found"

        unmatched = []
        for change in changes_with_patterns:
            pattern = change["search_pattern"]
            change_type = change["type"]

            # For removal patterns that match file paths, check paths
            if change_type == "removal" and "/" in pattern:
                path_matches = _match_pattern_against_paths(pattern, VANILLA_FILES)
                content_matches = _match_pattern_against_files(pattern, VANILLA_FILES)
                if not path_matches and not content_matches:
                    desc = change.get("removed", change.get("description", "?"))
                    unmatched.append(f"  {change_type}: {desc!r} — pattern: {pattern}")
            else:
                content_matches = _match_pattern_against_files(pattern, VANILLA_FILES)
                if not content_matches:
                    # Also try matching against file paths
                    path_matches = _match_pattern_against_paths(pattern, VANILLA_FILES)
                    if not path_matches:
                        desc = change.get(
                            "old_import",
                            change.get("removed", change.get("description", "?")),
                        )
                        unmatched.append(f"  {change_type}: {desc!r} — pattern: {pattern}")

        assert not unmatched, (
            f"{len(unmatched)} patterns did not match any vanilla v0.10 file:\n"
            + "\n".join(unmatched)
        )


# ===========================================================================
# Test 2: Search patterns match ALS-style v0.10 project
# ===========================================================================


class TestSearchPatternsMatchALSStyleV010:
    """Every search_pattern must match at least one file in the ALS-style project."""

    def test_all_search_patterns_have_matches(self, migration_doc):
        changes_with_patterns = _get_changes_with_search_pattern(migration_doc)
        assert len(changes_with_patterns) > 0

        unmatched = []
        for change in changes_with_patterns:
            pattern = change["search_pattern"]
            change_type = change["type"]

            if change_type == "removal" and "/" in pattern:
                path_matches = _match_pattern_against_paths(pattern, ALS_STYLE_FILES)
                content_matches = _match_pattern_against_files(pattern, ALS_STYLE_FILES)
                if not path_matches and not content_matches:
                    desc = change.get("removed", change.get("description", "?"))
                    unmatched.append(f"  {change_type}: {desc!r} — pattern: {pattern}")
            else:
                content_matches = _match_pattern_against_files(pattern, ALS_STYLE_FILES)
                if not content_matches:
                    path_matches = _match_pattern_against_paths(pattern, ALS_STYLE_FILES)
                    if not path_matches:
                        desc = change.get(
                            "old_import",
                            change.get("removed", change.get("description", "?")),
                        )
                        unmatched.append(f"  {change_type}: {desc!r} — pattern: {pattern}")

        assert not unmatched, (
            f"{len(unmatched)} patterns did not match any ALS-style v0.10 file:\n"
            + "\n".join(unmatched)
        )


# ===========================================================================
# Test 3: Patterns should NOT match in a v0.11 project
# ===========================================================================


class TestPatternsDoNotMatchV011:
    """import_change and removal patterns should NOT match in a migrated v0.11 project."""

    def test_import_changes_do_not_match(self, migration_doc):
        import_changes = _get_changes_by_type(migration_doc, "import_change")
        false_positives = []

        for change in import_changes:
            pattern = change["search_pattern"]
            matches = _match_pattern_against_files(pattern, V011_FILES)
            if matches:
                false_positives.append(
                    f"  import_change pattern {pattern!r} matched in v0.11: "
                    f"{matches[0][0]}:{matches[0][1]}"
                )

        assert not false_positives, (
            f"{len(false_positives)} import_change patterns still match in v0.11:\n"
            + "\n".join(false_positives)
        )

    def test_removal_patterns_do_not_match(self, migration_doc):
        removals = _get_changes_by_type(migration_doc, "removal")
        false_positives = []

        for change in removals:
            pattern = change.get("search_pattern")
            if not pattern:
                continue

            # Check against file content
            content_matches = _match_pattern_against_files(pattern, V011_FILES)
            # Check against file paths
            path_matches = _match_pattern_against_paths(pattern, V011_FILES)

            if content_matches or path_matches:
                desc = change.get("removed", "?")
                false_positives.append(
                    f"  removal pattern {pattern!r} ({desc}) still matches in v0.11"
                )

        assert not false_positives, (
            f"{len(false_positives)} removal patterns still match in v0.11:\n"
            + "\n".join(false_positives)
        )


# ===========================================================================
# Test 4: Facility-specific files are preserved
# ===========================================================================


class TestFacilitySpecificFilesPreserved:
    """Facility-specific files should NOT be matched by any removal pattern."""

    def test_live_monitoring_not_matched(self, migration_doc):
        """live_monitoring.py should not be matched by capability removal patterns."""
        removals = _get_changes_by_type(migration_doc, "removal")
        live_monitoring_content = FACILITY_SPECIFIC_FILES[
            "src/control_assistant/capabilities/live_monitoring.py"
        ]
        live_monitoring_path = "src/control_assistant/capabilities/live_monitoring.py"

        for change in removals:
            pattern = change.get("search_pattern")
            if not pattern:
                continue
            compiled = re.compile(pattern)

            # Should not match file path
            assert not compiled.search(live_monitoring_path), (
                f"Removal pattern {pattern!r} matched live_monitoring path"
            )

            # Should not match file content
            for line in live_monitoring_content.splitlines():
                # Only flag if the pattern seems to be targeting THIS file for removal
                # (import_change patterns may legitimately match — that's fine)
                if change["type"] == "removal" and compiled.search(line):
                    # Capability removal pattern should only match the 4 native capabilities
                    if "capabilities/" in pattern:
                        assert not compiled.search(live_monitoring_path), (
                            f"Removal pattern {pattern!r} matched live_monitoring.py path"
                        )

    def test_launcher_service_not_matched(self, migration_doc):
        """launcher/ service directory should not be matched by service removal patterns."""
        removals = _get_changes_by_type(migration_doc, "removal")

        for change in removals:
            pattern = change.get("search_pattern")
            if not pattern:
                continue
            compiled = re.compile(pattern)

            for path in FACILITY_SPECIFIC_FILES:
                if "launcher" in path:
                    assert not compiled.search(path), (
                        f"Removal pattern {pattern!r} matched facility-specific "
                        f"launcher file: {path}"
                    )

    def test_launcher_results_context_not_removed(self, migration_doc):
        """LAUNCHER_RESULTS ContextClassRegistration should not be targeted for removal."""
        removals = _get_changes_by_type(migration_doc, "removal")

        # The ContextClassRegistration removal pattern matches ALL occurrences,
        # but the migration_notes should clarify only 4 specific ones to remove.
        # Verify the pattern doesn't match LAUNCHER_RESULTS specifically.
        launcher_line = (
            "ContextClassRegistration(\n"
            '    context_type="LAUNCHER_RESULTS",\n'
            '    module_path="control_assistant.context_classes",\n'
            '    class_name="LauncherResultsContext"\n'
            ")"
        )

        for change in removals:
            pattern = change.get("search_pattern")
            if not pattern:
                continue
            compiled = re.compile(pattern)
            if compiled.search(launcher_line):
                # The generic pattern matches — that's expected.
                # Verify the migration_notes specify only the 4 native ones.
                notes = change.get("migration_notes", "")
                if "ContextClassRegistration" in change.get("removed", ""):
                    assert "LAUNCHER_RESULTS" not in notes, (
                        "migration_notes for ContextClassRegistration removal "
                        "should NOT list LAUNCHER_RESULTS"
                    )


# ===========================================================================
# Test 5: Schema compliance
# ===========================================================================


class TestSchemaCompliance:
    """Every change entry must comply with the migration document schema."""

    def test_required_top_level_fields(self, migration_doc):
        missing = REQUIRED_TOP_LEVEL - set(migration_doc.keys())
        assert not missing, f"Missing required top-level fields: {missing}"

    def test_each_change_has_type(self, migration_doc):
        for i, change in enumerate(migration_doc["changes"]):
            assert "type" in change, f"Change #{i} missing 'type' field"

    def test_change_types_are_known(self, migration_doc):
        for i, change in enumerate(migration_doc["changes"]):
            assert change["type"] in CHANGE_TYPE_REQUIRED, (
                f"Change #{i} has unknown type: {change['type']}"
            )

    def test_required_fields_per_type(self, migration_doc):
        missing_fields = []
        for i, change in enumerate(migration_doc["changes"]):
            change_type = change["type"]
            required = CHANGE_TYPE_REQUIRED.get(change_type, set())
            present = set(change.keys()) - {"type"}
            missing = required - present
            if missing:
                desc = change.get(
                    "old_import",
                    change.get("removed", change.get("description", "?")),
                )
                missing_fields.append(
                    f"  Change #{i} ({change_type}): missing {missing} — {desc!r}"
                )
        assert not missing_fields, (
            f"{len(missing_fields)} changes have missing required fields:\n"
            + "\n".join(missing_fields)
        )

    def test_valid_risk_levels(self, migration_doc):
        invalid = []
        for i, change in enumerate(migration_doc["changes"]):
            risk = change.get("risk")
            if risk is not None and risk not in VALID_RISK_LEVELS:
                invalid.append(f"  Change #{i}: risk={risk!r}")
        assert not invalid, "Invalid risk levels:\n" + "\n".join(invalid)

    def test_valid_automatable_values(self, migration_doc):
        invalid = []
        for i, change in enumerate(migration_doc["changes"]):
            auto = change.get("automatable")
            if auto is not None and auto not in VALID_AUTOMATABLE:
                invalid.append(f"  Change #{i}: automatable={auto!r}")
        assert not invalid, "Invalid automatable values:\n" + "\n".join(invalid)


# ===========================================================================
# Test 6: All search patterns are valid regex
# ===========================================================================


class TestSearchPatternsAreValidRegex:
    """Every search_pattern must compile without error."""

    def test_all_patterns_compile(self, migration_doc):
        failures = []
        for i, change in enumerate(migration_doc["changes"]):
            pattern = change.get("search_pattern")
            if pattern is None:
                continue
            try:
                re.compile(pattern)
            except re.error as e:
                desc = change.get(
                    "old_import",
                    change.get("removed", change.get("description", "?")),
                )
                failures.append(f"  Change #{i}: {desc!r} — {e}")
        assert not failures, f"{len(failures)} patterns failed to compile:\n" + "\n".join(failures)


# ===========================================================================
# Test 7: Validation commands are syntactically valid Python
# ===========================================================================


class TestValidationCommands:
    """Validation commands should execute successfully against the framework."""

    def test_validation_commands_exist(self, migration_doc):
        assert "validation" in migration_doc, "No validation section in migration doc"
        assert len(migration_doc["validation"]) > 0, "Empty validation section"

    def test_validation_commands_are_python_or_pytest(self, migration_doc):
        for entry in migration_doc["validation"]:
            cmd = entry["command"]
            assert cmd.startswith("python") or cmd.startswith("pytest"), (
                f"Unexpected command prefix: {cmd}"
            )

    @pytest.mark.parametrize(
        "idx",
        range(6),  # First 6 are python -c import checks
    )
    def test_python_import_validation_commands(self, migration_doc, idx):
        """Run the python -c import validation commands."""
        validations = migration_doc["validation"]
        if idx >= len(validations):
            pytest.skip("Not enough validation entries")

        entry = validations[idx]
        cmd = entry["command"]

        if not cmd.startswith("python -c"):
            pytest.skip("Not a python -c command")

        # Use sys.executable so the subprocess runs in the same venv as pytest,
        # rather than whatever "python" resolves to via PATH/pyenv.
        import subprocess
        import sys

        cmd_fixed = sys.executable + cmd[len("python") :]
        result = subprocess.run(cmd_fixed, shell=True, capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, (
            f"Validation command failed: {cmd}\nstderr: {result.stderr}\nstdout: {result.stdout}"
        )


# ===========================================================================
# Test 8: Relative import patterns specifically
# ===========================================================================


class TestRelativeImportPatterns:
    """Verify the relative import patterns added for ALS-style projects."""

    def test_relative_context_class_patterns_match(self, migration_doc):
        """Relative import patterns for context classes must match template-style imports."""
        relative_changes = [
            c
            for c in _get_changes_by_type(migration_doc, "import_change")
            if "from .." in c.get("old_import", "")
        ]
        assert len(relative_changes) > 0, "No relative import changes found"

        # These exact lines appear in template-generated capabilities
        expected_matches = [
            "from ..context_classes import ChannelAddressesContext",
            "from ..context_classes import ChannelValue, ChannelValuesContext",
            "from ..context_classes import ChannelWriteResult, ChannelWriteResultsContext, WriteVerificationInfo",
            "from ..context_classes import ArchiverDataContext",
            "from ..services.channel_finder.service import ChannelFinderService",
        ]

        for expected_line in expected_matches:
            matched_by = None
            for change in relative_changes:
                pattern = change["search_pattern"]
                if re.search(pattern, expected_line):
                    matched_by = pattern
                    break
            assert matched_by is not None, (
                f"Expected line not matched by any relative pattern: {expected_line!r}"
            )

    def test_relative_context_and_service_patterns_exist(self, migration_doc):
        """Relative import patterns exist for context classes and channel_finder service."""
        relative_changes = [
            c
            for c in _get_changes_by_type(migration_doc, "import_change")
            if c.get("old_import", "").startswith("from ..")
        ]
        # Should have: 4 context class patterns + 1 multi-import + 1 service pattern = 6
        assert len(relative_changes) >= 5, (
            f"Expected at least 5 relative import changes, got {len(relative_changes)}"
        )

    def test_helper_model_imports_covered(self, migration_doc):
        """ChannelValue, ChannelWriteResult, WriteVerificationInfo should be covered."""
        all_changes = _get_changes_with_search_pattern(migration_doc)

        helper_lines = [
            "from ..context_classes import ChannelValue, ChannelValuesContext",
            "from ..context_classes import ChannelWriteResult, ChannelWriteResultsContext, WriteVerificationInfo",
            "from test_facility.context_classes import ChannelValuesContext",
        ]

        for line in helper_lines:
            matched = False
            for change in all_changes:
                if re.search(change["search_pattern"], line):
                    matched = True
                    break
            assert matched, f"Helper model import line not covered: {line!r}"


# ===========================================================================
# Test 9: Exotic patterns from real ALS Assistant
# ===========================================================================


class TestExoticPatternsCovered:
    """Verify that exotic patterns from the real ALS Assistant are handled correctly.

    These tests validate that the migration doc patterns interact correctly with
    real-world code that has features like conditional imports, deferred imports,
    extra model fields, triple-dot relative imports, and sys.path.insert hacks.
    """

    def test_conditional_try_except_imports_not_targeted_for_removal(self, migration_doc):
        """Conditional try/except imports from osprey should NOT be matched by removal patterns.

        Real ALS has:
            try:
                from osprey.models import get_chat_completion
            except ImportError:
                get_chat_completion = None

        These import from osprey (the framework), not the downstream package, so
        no migration pattern should target them for removal.
        """
        removals = _get_changes_by_type(migration_doc, "removal")
        conditional_lines = [
            "from osprey.models import get_chat_completion",
            "from osprey.registry import get_registry",
            "from osprey.utils.config import get_config_value",
        ]

        for line in conditional_lines:
            for change in removals:
                pattern = change.get("search_pattern")
                if not pattern:
                    continue
                assert not re.search(pattern, line), (
                    f"Removal pattern {pattern!r} incorrectly matches osprey import: {line!r}"
                )

    def test_deferred_imports_inside_methods_not_targeted(self, migration_doc):
        """Deferred imports inside method bodies that import from osprey should be safe.

        Real ALS has inside execute():
            from osprey.utils.config import get_config_value
            from osprey.registry import get_registry
        """
        all_changes = _get_changes_with_search_pattern(migration_doc)
        deferred_lines = [
            "from osprey.registry import get_registry",
            "from osprey.utils.config import get_config_value",
            "from osprey.approval.approval_manager import get_python_execution_evaluator",
        ]

        for line in deferred_lines:
            for change in all_changes:
                if change["type"] in ("removal", "import_change"):
                    pattern = change["search_pattern"]
                    assert not re.search(pattern, line), (
                        f"Pattern {pattern!r} ({change['type']}) incorrectly matches "
                        f"osprey deferred import: {line!r}"
                    )

    def test_capabilities_init_reexports_matched_when_present(self, migration_doc):
        """capabilities/__init__.py re-exports ARE matched by capability import patterns.

        Vanilla fixture has re-exports like:
            from test_facility.capabilities.channel_finding import ChannelFindingCapability
        These should be matched by the capability import_change patterns.
        """
        import_changes = _get_changes_by_type(migration_doc, "import_change")
        cap_import_patterns = [
            c
            for c in import_changes
            if "capabilities.channel_" in c.get("old_import", "")
            or "capabilities.archiver_" in c.get("old_import", "")
        ]

        reexport_lines = [
            "from test_facility.capabilities.channel_finding import ChannelFindingCapability",
            "from test_facility.capabilities.channel_read import ChannelReadCapability",
            "from test_facility.capabilities.channel_write import ChannelWriteCapability",
            "from test_facility.capabilities.archiver_retrieval import ArchiverRetrievalCapability",
        ]

        for line in reexport_lines:
            matched = any(re.search(c["search_pattern"], line) for c in cap_import_patterns)
            assert matched, (
                f"Capability re-export not matched by any import_change pattern: {line!r}"
            )

    def test_triple_dot_relative_imports_covered_by_service_removal(self, migration_doc):
        """Triple-dot relative imports within services/channel_finder/ are covered.

        Real ALS has in pipelines/hierarchical/pipeline.py:
            from ...core.base_pipeline import BasePipeline
            from ...core.models import ChannelFinderResult

        These files live inside services/channel_finder/ which is deleted entirely,
        so the services/channel_finder/ removal pattern covers them via path matching.
        """
        removals = _get_changes_by_type(migration_doc, "removal")
        service_removal = [
            c for c in removals if "services/channel_finder/" in c.get("removed", "")
        ]
        assert len(service_removal) > 0, "No services/channel_finder/ removal found"

        triple_dot_paths = [
            "src/control_assistant/services/channel_finder/pipelines/hierarchical/pipeline.py",
            "src/control_assistant/services/channel_finder/pipelines/in_context/pipeline.py",
            "src/control_assistant/services/channel_finder/examples/custom_pipeline_example.py",
        ]

        for path in triple_dot_paths:
            matched = any(
                re.search(c["search_pattern"], path)
                for c in service_removal
                if c.get("search_pattern")
            )
            assert matched, f"Triple-dot import file path not covered by service removal: {path!r}"

    def test_sys_path_insert_in_data_tools_covered(self, migration_doc):
        """sys.path.insert hacks in data/tools/ are covered by the removal pattern.

        Real ALS has in every data/tools/*.py:
            sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

        These files are deleted entirely by the data/tools/ removal pattern.
        """
        removals = _get_changes_by_type(migration_doc, "removal")
        data_tools_removal = [c for c in removals if "data/tools/" in c.get("removed", "")]
        assert len(data_tools_removal) > 0, "No data/tools/ removal found"

        tool_files = [
            "src/control_assistant/data/tools/build_channel_database.py",
            "src/control_assistant/data/tools/validate_database.py",
            "src/control_assistant/data/tools/preview_database.py",
            "src/control_assistant/data/tools/llm_channel_namer.py",
        ]

        for path in tool_files:
            matched = any(
                re.search(c["search_pattern"], path)
                for c in data_tools_removal
                if c.get("search_pattern")
            )
            assert matched, f"data/tools/ file not covered by removal pattern: {path!r}"

    def test_sys_path_insert_in_benchmarks_cli_covered(self, migration_doc):
        """sys.path.insert in benchmarks/cli.py is covered by service removal.

        Real ALS has in services/channel_finder/benchmarks/cli.py:
            sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

        This file lives inside services/channel_finder/ which is deleted entirely.
        """
        removals = _get_changes_by_type(migration_doc, "removal")
        service_removal = [
            c for c in removals if "services/channel_finder/" in c.get("removed", "")
        ]

        benchmarks_path = "src/control_assistant/services/channel_finder/benchmarks/cli.py"
        matched = any(
            re.search(c["search_pattern"], benchmarks_path)
            for c in service_removal
            if c.get("search_pattern")
        )
        assert matched, f"benchmarks/cli.py not covered by service removal: {benchmarks_path!r}"

    def test_description_field_on_capability_registration_still_matches(self, migration_doc):
        """CapabilityRegistration with description= field should still be matched.

        Real ALS uses:
            CapabilityRegistration(
                name="channel_finding",
                ...
                description="Find control system channels using semantic search",
                ...
            )

        The CapabilityRegistration removal pattern should match regardless of
        whether description= is present.
        """
        removals = _get_changes_by_type(migration_doc, "removal")
        cap_reg_removals = [c for c in removals if "CapabilityRegistration" in c.get("removed", "")]
        assert len(cap_reg_removals) > 0, "No CapabilityRegistration removal found"

        # The line that starts the registration block
        reg_line = "                CapabilityRegistration("
        matched = any(re.search(c["search_pattern"], reg_line) for c in cap_reg_removals)
        assert matched, "CapabilityRegistration( not matched by removal pattern"

    def test_examples_directory_covered_by_service_removal(self, migration_doc):
        """services/channel_finder/examples/ directory is covered by service removal.

        Real ALS has extra files:
            services/channel_finder/examples/custom_database_example.py
            services/channel_finder/examples/custom_pipeline_example.py
        """
        removals = _get_changes_by_type(migration_doc, "removal")
        service_removal = [
            c for c in removals if "services/channel_finder/" in c.get("removed", "")
        ]

        example_paths = [
            "src/control_assistant/services/channel_finder/examples/custom_database_example.py",
            "src/control_assistant/services/channel_finder/examples/custom_pipeline_example.py",
        ]

        for path in example_paths:
            matched = any(
                re.search(c["search_pattern"], path)
                for c in service_removal
                if c.get("search_pattern")
            )
            assert matched, f"examples/ file not covered by service removal: {path!r}"

    def test_empty_capabilities_init_not_false_positive(self, migration_doc):
        """An empty capabilities/__init__.py (docstring only) should not trigger
        import_change patterns — there's nothing to migrate in it."""
        import_changes = _get_changes_by_type(migration_doc, "import_change")
        empty_init = ALS_STYLE_FILES["src/control_assistant/capabilities/__init__.py"]

        false_positives = []
        for change in import_changes:
            pattern = change["search_pattern"]
            for line in empty_init.splitlines():
                if re.search(pattern, line):
                    false_positives.append(
                        f"  {change.get('old_import', '?')} — pattern {pattern!r} matched: {line!r}"
                    )

        assert not false_positives, (
            "import_change patterns matched empty capabilities/__init__.py:\n"
            + "\n".join(false_positives)
        )
