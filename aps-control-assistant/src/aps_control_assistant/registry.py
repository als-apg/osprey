"""
Component registry for aps-control-assistant.

Registers all capabilities and context classes with the Osprey framework.
"""

from osprey.registry import (
    RegistryConfigProvider,
    extend_framework_registry,
    CapabilityRegistration,
    ContextClassRegistration,
    RegistryConfig
)


class ApsControlAssistantRegistryProvider(RegistryConfigProvider):
    """Registry provider for aps-control-assistant.

    Registers control system integration capabilities:
    - Channel finding (semantic search for control system channels)
    - Channel value retrieval (read current values from channels)
    - Archiver retrieval (query historical time-series data)

    Based on production patterns from ALS Assistant.
    """

    def get_registry_config(self) -> RegistryConfig:
        """Return registry configuration for aps-control-assistant."""
        return extend_framework_registry(
            capabilities=[
                CapabilityRegistration(
                    name="channel_finding",
                    module_path="aps_control_assistant.capabilities.channel_finding",
                    class_name="ChannelFindingCapability",
                    description="Find control system channels using semantic search",
                    provides=["CHANNEL_ADDRESSES"],
                    requires=[]
                ),
                CapabilityRegistration(
                    name="channel_value_retrieval",
                    module_path="aps_control_assistant.capabilities.channel_value_retrieval",
                    class_name="ChannelValueRetrievalCapability",
                    description="Retrieve current values from control system channels",
                    provides=["CHANNEL_VALUES"],
                    requires=["CHANNEL_ADDRESSES"]
                ),
                CapabilityRegistration(
                    name="archiver_retrieval",
                    module_path="aps_control_assistant.capabilities.archiver_retrieval",
                    class_name="ArchiverRetrievalCapability",
                    description="Query historical time-series data from the archiver",
                    provides=["ARCHIVER_DATA"],
                    requires=["CHANNEL_ADDRESSES", "TIME_RANGE"]
                ),
            ],
            context_classes=[
                ContextClassRegistration(
                    context_type="CHANNEL_ADDRESSES",
                    module_path="aps_control_assistant.context_classes",
                    class_name="ChannelAddressesContext"
                ),
                ContextClassRegistration(
                    context_type="CHANNEL_VALUES",
                    module_path="aps_control_assistant.context_classes",
                    class_name="ChannelValuesContext"
                ),
                ContextClassRegistration(
                    context_type="ARCHIVER_DATA",
                    module_path="aps_control_assistant.context_classes",
                    class_name="ArchiverDataContext"
                ),
            ]
        )

