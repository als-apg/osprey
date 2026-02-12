"""LLM-based Channel Name Generator.

This module provides intelligent channel name generation using language models.
It takes short technical names and natural language descriptions to create
descriptive, intuitive channel names that are self-documenting.

Key Features:
- Batch processing for efficiency
- Configurable LLM providers
- Validation and quality checks
"""

import logging
from collections import defaultdict

from pydantic import BaseModel, Field
from tqdm import tqdm

from osprey.models.completion import get_chat_completion

logger = logging.getLogger(__name__)


class ChannelNames(BaseModel):
    """Structured output model for generated channel names."""

    names: list[str] = Field(
        description="List of generated PascalCase channel names, one for each input channel in order"
    )


class LLMChannelNamer:
    """Generate descriptive channel names using language models."""

    def __init__(
        self,
        provider: str = "cborg",
        model_id: str = "google/gemini-flash",
        max_tokens: int = 1000,
        batch_size: int = 10,
        base_url: str | None = None,
        api_key: str | None = None,
    ):
        """Initialize the LLM channel namer.

        Args:
            provider: LLM provider ('cborg', 'anthropic', 'openai')
            model_id: Model identifier
            max_tokens: Maximum tokens per request
            batch_size: Number of channels to process per batch
            base_url: API base URL (optional, from config if not provided)
            api_key: API key (optional, from config/env if not provided)
        """
        self.provider = provider
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.batch_size = batch_size
        self.base_url = base_url
        self.api_key = api_key

    def _create_prompt_for_batch(self, channels: list[dict]) -> str:
        """Create a prompt for a batch of channels.

        Args:
            channels: List of dicts with 'short_name' and 'description'

        Returns:
            Formatted prompt string
        """
        prompt = (
            "Generate descriptive PascalCase channel names for particle accelerator "
            "control channels.\n\n"
            "CRITICAL NAMING RULES:\n"
            "- PascalCase format (e.g., AcceleratingTubeTopSteeringCoilXSetPoint)\n"
            "- 2-7 words, prioritize uniqueness and clarity\n"
            '- **ALWAYS use "SetPoint" (not "Set") for writable/settable values**\n'
            '- **ALWAYS use "ReadBack" (not "RB") for read-only/readback values**\n'
            "- **CRITICAL: ALWAYS include specific location/position information "
            "to ensure uniqueness**\n"
            "- Avoid unnecessary abbreviations (use full words)\n\n"
            "LOCATION/POSITION PRIORITY:\n"
            "- **MANDATORY:** Extract and include ALL location/position details "
            "from descriptions:\n"
            '  * Specific locations: "accelerating tube", "decelerating tube", '
            '"terminal", "beamline1", etc.\n'
            '  * Positions within locations: "top", "bottom", "beginning", '
            '"end", "middle"\n'
            '  * Direction/axis: "X direction", "Y direction", "horizontal", '
            '"vertical"\n'
            '  * Other identifying details: "inside tank", specific device IDs, etc.\n'
            "- Names MUST be unique - different physical locations = different names\n"
            "- Pattern: [Location][Position][Device][Axis][Property][SetPoint/ReadBack]\n\n"
            "SETPOINT vs READBACK GUIDELINES:\n"
            '- If description mentions "set value", "set point", "settable", '
            '"writable" \u2192 use "SetPoint"\n'
            '- If description mentions "read back", "read only", "readback", '
            '"measured" \u2192 use "ReadBack"\n'
            '- Be consistent: ALWAYS use full words "SetPoint" and "ReadBack" '
            "for database consistency\n\n"
            "EXAMPLES showing location specificity:\n"
            '1. Short: "SX3Set", Description: "Steering coil current tilting beam in '
            'X direction inside accelerating tube (top position, inside tank), set value"\n'
            "   \u2192 AcceleratingTubeTopSteeringCoilXSetPoint\n\n"
            '2. Short: "SX40Set", Description: "Steering coil current tilting beam in '
            'X direction inside accelerating tube (bottom position, inside tank), set value"\n'
            "   \u2192 AcceleratingTubeBottomSteeringCoilXSetPoint\n\n"
            '3. Short: "IP149APressure", Description: "Pressure at IP149A ion pump '
            "located in the beginning of HFEL beamline (FEL output beamline) "
            'measured in Tor"\n'
            "   \u2192 HFELBeamLineBeginningIonPumpPressure\n\n"
            '4. Short: "FansVoltage", Description: "Voltage of fans cooling terminal '
            'electronics, only read back value"\n'
            "   \u2192 TerminalCoolingFanVoltageReadBack\n\n"
            '5. Short: "I_mtrs", Description: "Diagnostic current through grading '
            "resistor chain in decelerating tube (monitors voltage distribution "
            'health)"\n'
            "   \u2192 DeceleratingTubeGradingResistorDiagnosticCurrent\n\n"
            f"INPUT CHANNELS (generate EXACTLY {len(channels)} names in order):\n"
        )

        for i, ch in enumerate(channels, 1):
            prompt += f'\n{i}. Short: "{ch["short_name"]}"\n'
            prompt += f'   Description: "{ch["description"]}"\n'

        return prompt

    def generate_names_batch(self, channels: list[dict]) -> list[str]:
        """Generate names for a batch of channels using LLM with structured output.

        Args:
            channels: List of dicts with 'short_name' and 'description'

        Returns:
            List of generated channel names (same order as input)
        """
        if not channels:
            return []

        prompt = self._create_prompt_for_batch(channels)

        try:
            provider_config = {}
            if self.base_url:
                provider_config["base_url"] = self.base_url
            if self.api_key:
                provider_config["api_key"] = self.api_key

            result = get_chat_completion(
                message=prompt,
                provider=self.provider,
                model_id=self.model_id,
                max_tokens=self.max_tokens,
                base_url=self.base_url,
                provider_config=provider_config if provider_config else None,
                output_model=ChannelNames,
            )

            if isinstance(result, dict):
                names = result.get("names", [])
            else:
                names = result.names

            if len(names) != len(channels):
                logger.error(f"Expected {len(channels)} names, got {len(names)}.")
                raise ValueError(
                    f"Wrong number of names: expected {len(channels)}, got {len(names)}"
                )

            validated_names = []
            for i, name in enumerate(names):
                if self._is_valid_channel_name(name):
                    validated_names.append(name)
                else:
                    logger.warning(f"Invalid generated name '{name}', using original short name")
                    validated_names.append(channels[i]["short_name"])

            return validated_names

        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            logger.info("Returning original short names")
            return [ch["short_name"] for ch in channels]

    def _is_valid_channel_name(self, name: str) -> bool:
        """Check if a generated name is valid."""
        if not name or len(name) < 3:
            return False
        if not name.replace("_", "").isalnum():
            return False
        if not name[0].isupper():
            return False
        if len(name) > 80:
            return False
        return True

    def _create_duplicate_resolution_prompt(self, duplicate_groups: dict[str, list[dict]]) -> str:
        """Create a prompt to resolve duplicate channel names."""
        prompt = (
            "DUPLICATE NAME RESOLUTION: Generate MORE SPECIFIC PascalCase names "
            "to resolve duplicates.\n\n"
            "SITUATION: Multiple channels were assigned the same name. They need "
            "MORE SPECIFIC names that highlight their differences.\n\n"
            "RESOLUTION STRATEGY:\n"
            "- Include MORE location/position details\n"
            "- Emphasize what makes each channel UNIQUE\n"
            "- Use the FULL location hierarchy if needed\n"
            "- Pattern: [DetailedLocation][SpecificPosition][Device][Axis][Property]"
            "[SetPoint/ReadBack]\n\n"
        )
        total_channels = sum(len(channels) for channels in duplicate_groups.values())
        prompt += f"GENERATE EXACTLY {total_channels} UNIQUE NAMES:\n\n"

        channel_num = 1
        for dup_name, channels in duplicate_groups.items():
            prompt += (
                f"=== DUPLICATE GROUP: '{dup_name}' (needs {len(channels)} distinct names) ===\n"
            )
            for ch in channels:
                prompt += f'{channel_num}. Short: "{ch["short_name"]}"\n'
                prompt += f'   Description: "{ch["description"]}"\n'
                prompt += f'   Previous Name (TOO GENERIC): "{ch["original_name"]}"\n\n'
                channel_num += 1

        return prompt

    def resolve_duplicates(self, channels: list[dict], names: list[str]) -> list[str]:
        """Check for duplicate names and regenerate them with more specificity.

        Args:
            channels: Original channel dicts with 'short_name' and 'description'
            names: Generated names (same length as channels)

        Returns:
            List of names with duplicates resolved
        """
        name_to_indices = defaultdict(list)
        for idx, name in enumerate(names):
            name_to_indices[name].append(idx)

        duplicates = {
            name: indices for name, indices in name_to_indices.items() if len(indices) > 1
        }

        if not duplicates:
            return names

        total_dups = sum(len(indices) for indices in duplicates.values())
        print(
            f"\n\u26a0\ufe0f  Found {len(duplicates)} duplicate name(s) "
            f"affecting {total_dups} channels:"
        )
        for dup_name, indices in duplicates.items():
            print(f"   '{dup_name}': {len(indices)} channels")

        print("\U0001f504 Regenerating names with more specificity...")

        duplicate_groups = {}
        for dup_name, indices in duplicates.items():
            duplicate_groups[dup_name] = [
                {
                    "short_name": channels[idx]["short_name"],
                    "description": channels[idx]["description"],
                    "original_name": dup_name,
                }
                for idx in indices
            ]

        prompt = self._create_duplicate_resolution_prompt(duplicate_groups)

        try:
            provider_config = {}
            if self.base_url:
                provider_config["base_url"] = self.base_url
            if self.api_key:
                provider_config["api_key"] = self.api_key

            response = get_chat_completion(
                message=prompt,
                provider=self.provider,
                model_id=self.model_id,
                max_tokens=self.max_tokens,
                temperature=0.3,
                base_url=self.base_url,
                provider_config=provider_config if provider_config else None,
                output_model=ChannelNames,
            )

            new_names = response.names

            result = names.copy()
            name_idx = 0
            for _dup_name, indices in duplicates.items():
                for idx in indices:
                    if name_idx < len(new_names):
                        result[idx] = new_names[name_idx]
                        name_idx += 1

            print("   \u2713 Resolved all duplicates")
            return result

        except Exception as e:
            print(f"   \u26a0\ufe0f  Failed to resolve duplicates: {e}")
            print("   Keeping original names (with duplicates)")
            return names

    def generate_names(self, channels: list[dict]) -> list[str]:
        """Generate names for all channels with batching and progress bar.

        Args:
            channels: List of dicts with 'short_name' and 'description'

        Returns:
            List of generated channel names (same order as input)
        """
        all_names = []

        with tqdm(total=len(channels), desc="Generating channel names", unit="channel") as pbar:
            for i in range(0, len(channels), self.batch_size):
                batch = channels[i : i + self.batch_size]

                batch_names = self.generate_names_batch(batch)
                all_names.extend(batch_names)

                pbar.update(len(batch))

        all_names = self.resolve_duplicates(channels, all_names)

        return all_names


def create_namer_from_config(config_path: str | None = None) -> LLMChannelNamer:
    """Create an LLM channel namer from configuration file.

    Args:
        config_path: Path to config.yml file (optional, uses default if not provided)

    Returns:
        Configured LLMChannelNamer instance
    """
    from osprey.services.channel_finder.utils.config import get_config, load_config

    if config_path:
        config = load_config(str(config_path))
    else:
        config = get_config()

    name_gen_config = config.get("channel_finder", {}).get("channel_name_generation", {})

    llm_config = name_gen_config.get("llm_model", {})
    provider = llm_config.get("provider", "cborg")

    api_config = config.get("api", {}).get("providers", {}).get(provider, {})
    base_url = api_config.get("base_url")
    api_key = api_config.get("api_key")

    return LLMChannelNamer(
        provider=provider,
        model_id=llm_config.get("model_id", "anthropic/claude-haiku"),
        max_tokens=llm_config.get("max_tokens", 1000),
        batch_size=name_gen_config.get("llm_batch_size", 10),
        base_url=base_url,
        api_key=api_key,
    )
