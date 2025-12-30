# Osprey GUI Configuration Guide

This guide explains how to configure the Osprey PyQt GUI for your facility.

## Quick Start

1. **Copy the example configuration file:**
   ```bash
   cd osprey/src/osprey/interfaces/pyqt
   cp gui_config.yml.example gui_config.yml
   ```

2. **Set up environment variables for API keys:**
   
   Create or update your `.env` file in the project root with your API key(s):
   ```bash
   # Choose ONE provider that your facility uses
   ANTHROPIC_API_KEY=your-anthropic-key-here
   # OR
   OPENAI_API_KEY=your-openai-key-here
   # OR
   CBORG_API_KEY=your-cborg-key-here
   # OR
   ARGO_API_KEY=your-argo-key-here
   # etc.
   ```

3. **Edit `gui_config.yml` to configure your LLM provider:**
   
   Update the `models.classifier` section with your facility's provider:
   ```yaml
   models:
     classifier:
       provider: anthropic  # Change to your provider
       model_id: claude-3-5-haiku-20241022  # Change to your model
       max_tokens: 4096
   ```

4. **Verify your provider is configured in the `api.providers` section:**
   
   Make sure the provider you chose has the correct `base_url` and uses the environment variable for the API key:
   ```yaml
   api:
     providers:
       anthropic:
         api_key: ${ANTHROPIC_API_KEY}
         base_url: https://api.anthropic.com
   ```

5. **Launch the GUI:**
   ```bash
   osprey gui
   ```

## Configuration Sections

### Required Configuration

#### 1. LLM Models (`models`)

The GUI requires at least the `classifier` model to be configured for routing queries between projects:

```yaml
models:
  classifier:
    provider: anthropic  # Your facility's LLM provider
    model_id: claude-3-5-haiku-20241022
    max_tokens: 4096
  
  # Optional: Orchestrator model for multi-project queries
  orchestrator:
    provider: anthropic
    model_id: claude-3-5-haiku-20241022
    max_tokens: 4096
```

**Supported Providers:**
- `anthropic` - Claude models (Recommended: Claude Haiku 3.5)
- `openai` - GPT models
- `argo` - Argonne National Laboratory service
- `cborg` - LBNL institutional provider
- `stanford` - Stanford AI Playground
- `google` - Google Gemini models
- `ollama` - Local models

#### 2. API Provider Configuration (`api.providers`)

Configure the API endpoint and credentials for your chosen provider:

```yaml
api:
  providers:
    anthropic:
      api_key: ${ANTHROPIC_API_KEY}  # Use environment variable
      base_url: https://api.anthropic.com
```

**Important:** Always use environment variables (`${VAR_NAME}`) for API keys. Never hardcode credentials in the config file.

### Optional Configuration

#### GUI Settings (`gui`)

Control GUI behavior and conversation management:

```yaml
gui:
  use_persistent_conversations: true
  conversation_storage_mode: json  # or 'postgresql'
  redirect_output_to_gui: true
  suppress_terminal_output: false
  group_system_messages: true
  enable_routing_feedback: true
```

**Conversation Storage Modes:**
- `json` - File-based storage (default, simple setup)
  - Messages stored in `osprey/src/osprey/_gui_data/conversations/`
  - Lightweight, portable, no database required
- `postgresql` - Database storage (requires PostgreSQL setup)
  - Requires `postgresql_uri` configuration
  - Better for large-scale deployments

**GUI Behavior Settings:**
- `redirect_output_to_gui` - Show framework logs in System Information tab
- `suppress_terminal_output` - Hide terminal output (GUI only)
- `group_system_messages` - Organize system messages by type in collapsible sections
- `enable_routing_feedback` - Prompt for feedback after routing decisions

For PostgreSQL storage, add:
```yaml
gui:
  conversation_storage_mode: postgresql
  postgresql_uri: postgresql://user:password@localhost:5432/osprey_gui
```

#### Routing Configuration (`routing`)

Advanced settings for multi-project routing and orchestration:

```yaml
routing:
  # Query caching for faster repeated queries
  cache:
    enabled: true
    max_size: 100
    ttl_seconds: 3600.0  # 1 hour
    similarity_threshold: 0.85
  
  # Advanced cache invalidation strategies
  advanced_invalidation:
    enabled: true
    adaptive_ttl: true  # Hot entries cached longer
    probabilistic_expiration: true  # Prevent cache stampede
    event_driven: true  # Auto-invalidate on config changes
  
  # Semantic analysis for better routing decisions
  semantic_analysis:
    enabled: true
    similarity_threshold: 0.5
    topic_similarity_threshold: 0.6
    max_context_history: 20
  
  # Multi-project orchestration
  orchestration:
    max_parallel: 3  # Max parallel sub-queries
  
  # Analytics and metrics
  analytics:
    max_history: 1000  # Max routing decisions to track
  
  # User feedback collection
  feedback:
    enabled: true
```

**Routing Features:**
- **Cache** - Speeds up repeated queries by caching routing decisions
- **Semantic Analysis** - Understands conversation context for better routing
- **Orchestration** - Handles queries requiring multiple projects
- **Analytics** - Tracks routing performance and accuracy
- **Feedback** - Collects user feedback to improve routing

#### Agent Control (`execution_control`)

Control agent capabilities and execution limits:

```yaml
execution_control:
  agent_control:
    task_extraction_bypass_enabled: false
    capability_selection_bypass_enabled: false
  
  epics:
    writes_enabled: false  # Set true to allow EPICS writes
  
  limits:
    max_reclassifications: 1
    max_planning_attempts: 2
    max_step_retries: 0
    max_execution_time_seconds: 300
    max_concurrent_classifications: 5
```

**Agent Control Options:**
- `task_extraction_bypass_enabled` - Skip task extraction (use full context)
- `capability_selection_bypass_enabled` - Activate all capabilities
- `epics.writes_enabled` - Allow EPICS control system writes

**Execution Limits:**
- `max_reclassifications` - Max task reclassification attempts
- `max_planning_attempts` - Max planning retries
- `max_step_retries` - Max retries per execution step
- `max_execution_time_seconds` - Total execution timeout
- `max_concurrent_classifications` - Parallel LLM classification limit

#### Approval Settings (`approval`)

Configure human approval requirements for sensitive operations:

```yaml
approval:
  global_mode: selective  # 'all', 'selective', or 'none'
  
  capabilities:
    python_execution:
      enabled: true
      mode: all_code  # 'all_code', 'epics_writes', or 'none'
    
    memory:
      enabled: true
```

**Global Modes:**
- `all` - Approve all operations
- `selective` - Use capability-specific settings
- `none` - No approval required

**Python Execution Modes:**
- `all_code` - Approve all code execution
- `epics_writes` - Approve only EPICS write operations
- `none` - No approval needed

#### Development Settings (`development`)

Settings for debugging and troubleshooting:

```yaml
development:
  debug: false  # Enable DEBUG logging level
  verbose_logging: false  # Enable verbose logging output
  raise_raw_errors: false  # Show full error stack traces
  
  prompts:
    print_all: false  # Save prompts to files
    show_all: false  # Display prompts in console
    latest_only: true  # Use latest.md instead of timestamps
  
  memory_monitor:
    enabled: true
    warning_threshold_mb: 500
    critical_threshold_mb: 1000
    check_interval_seconds: 5
```

**Debug Features:**
- `debug` - Enables DEBUG logging level (shows all framework messages)
- `verbose_logging` - Enables verbose logging output (development.verbose_logging)
- `raise_raw_errors` - Re-raises original exceptions for debugging
- `prompts.print_all` - Saves all prompts to `prompts/` directory
- `prompts.show_all` - Displays prompts in System Information tab

**Memory Monitoring:**
- `memory_monitor.enabled` - Enable automatic memory monitoring
- `memory_monitor.warning_threshold_mb` - Warning threshold in MB (default: 500)
- `memory_monitor.critical_threshold_mb` - Critical threshold in MB (default: 1000)
- `memory_monitor.check_interval_seconds` - Update interval in seconds (default: 5)

## Common Configurations

### For Anthropic (Claude)

```yaml
models:
  classifier:
    provider: anthropic
    model_id: claude-3-5-haiku-20241022
    max_tokens: 4096

api:
  providers:
    anthropic:
      api_key: ${ANTHROPIC_API_KEY}
      base_url: https://api.anthropic.com
```

Environment variable:
```bash
ANTHROPIC_API_KEY=sk-ant-...
```

### For OpenAI (GPT)

```yaml
models:
  classifier:
    provider: openai
    model_id: gpt-4o-mini
    max_tokens: 4096

api:
  providers:
    openai:
      api_key: ${OPENAI_API_KEY}
      base_url: https://api.openai.com/v1
```

Environment variable:
```bash
OPENAI_API_KEY=sk-...
```

### For Argonne (Argo)

```yaml
models:
  classifier:
    provider: argo
    model_id: gpt5
    max_tokens: 4096

api:
  providers:
    argo:
      api_key: ${ARGO_API_KEY}
      base_url: https://argo-bridge.cels.anl.gov/v1
```

Environment variable:
```bash
ARGO_API_KEY=your-argo-key
```

### For Local Ollama

```yaml
models:
  classifier:
    provider: ollama
    model_id: llama3.1
    max_tokens: 4096

api:
  providers:
    ollama:
      api_key: ollama  # Not used
      base_url: http://localhost:11434
      host: localhost
      port: 11434
```

No environment variable needed for Ollama.

## GUI Features

### Multi-Project Support

The GUI automatically discovers and loads all Osprey projects in your workspace:

- **Project Discovery** - Scans for projects with `config.yml` files
- **Automatic Routing** - Routes queries to the most appropriate project
- **Manual Selection** - Override automatic routing via Project Control panel
- **Project Enable/Disable** - Control which projects are available for routing

### Conversation Management

- **Persistent History** - Conversations saved automatically (JSON or PostgreSQL)
- **Multiple Conversations** - Create and switch between conversations
- **Message Search** - Find messages across conversations
- **Export/Import** - Backup and restore conversation history

### Advanced Routing

- **Intelligent Caching** - Speeds up repeated queries
- **Context Awareness** - Understands conversation flow
- **Multi-Project Orchestration** - Handles queries requiring multiple projects
- **User Feedback** - Learn from corrections to improve routing

### Monitoring & Analytics

- **System Information** - Real-time framework logs with color coding
- **LLM Details** - Track all LLM API calls and responses
- **Tool Usage** - Monitor capability execution and reasoning
- **Memory Monitoring** - Track framework memory usage
- **Analytics Dashboard** - Routing performance metrics

## Settings Dialog

Access via **Settings → Framework Settings** menu:

### Tabs:
1. **Agent Control** - Planning mode, EPICS writes, bypass options
2. **Approval** - Approval modes for sensitive operations
3. **Execution Limits** - Timeouts, retries, concurrency limits
4. **GUI Settings** - Conversation storage, output redirection, feedback
5. **Development/Debug** - Debug mode, logging, prompt visibility, memory monitoring
6. **Advanced Routing** - Cache, semantic analysis, orchestration settings

**Note:** Settings are saved to `gui_config.yml` and persist across restarts.

## Troubleshooting

### "GUI configuration file not found"

Make sure you've copied `gui_config.yml.example` to `gui_config.yml`:
```bash
cp gui_config.yml.example gui_config.yml
```

### "classifier model not configured"

Ensure your `gui_config.yml` has a `models.classifier` section with both `provider` and `model_id` specified.

### "Environment variable not set"

Make sure you've set the appropriate environment variable for your provider:
```bash
export ANTHROPIC_API_KEY=your-key-here
# or add to your .env file
```

### "Provider not configured"

Verify that the provider specified in `models.classifier.provider` has a corresponding entry in `api.providers`.

### Routing Issues

- **Clear Cache** - Use "🗑️ Clear Cache" button in Project Control panel
- **Check Projects** - Verify projects are enabled in Discovered Projects tab
- **Review Feedback** - Check if routing feedback is enabled and working
- **Analytics** - Review routing decisions in Analytics tab

### Performance Issues

- **Disable Debug Mode** - Turn off debug logging in Settings
- **Reduce Cache Size** - Lower `cache_max_size` in routing config
- **Limit Context History** - Reduce `max_context_history` setting
- **Disable Analytics** - Turn off analytics if not needed

## Security Best Practices

1. **Never commit `gui_config.yml`** - It's already in `.gitignore`
2. **Use environment variables** for all API keys
3. **Keep `gui_config.yml.example` updated** as a template for others
4. **Rotate API keys regularly** according to your facility's security policy
5. **Use read-only API keys** when possible for testing
6. **Review approval settings** - Enable approval for sensitive operations
7. **Monitor EPICS writes** - Keep `epics_writes_enabled: false` unless needed

## File Locations

- **Example template:** `osprey/src/osprey/interfaces/pyqt/gui_config.yml.example`
- **Your config:** `osprey/src/osprey/interfaces/pyqt/gui_config.yml` (create this)
- **Environment variables:** Project root `.env` file or shell environment
- **GUI runtime data:** `osprey/src/osprey/_gui_data/` (auto-created at runtime)
  - `conversations/` - Conversation history (JSON mode)
  - `routing_feedback.json` - User routing feedback
  - `routing_analytics.json` - Routing performance metrics
- **Prompt files (if enabled):** `prompts/` directory
- **Log files:** Check framework configuration for log directory

**Note:** The `_gui_data/` directory is automatically created by the GUI at runtime and contains user-specific data. It should be added to `.gitignore` and is safe to delete (will be recreated on next launch).

## Advanced Topics

### Custom Model Preferences

Override models for specific projects via the GUI:
1. Go to **Discovered Projects** tab
2. Click **Configure** button for a project
3. Set model overrides for infrastructure steps (classifier, orchestrator, etc.)

### PostgreSQL Setup

For PostgreSQL conversation storage:
1. Install PostgreSQL and create database
2. Set `conversation_storage_mode: postgresql` in config
3. Add `postgresql_uri` with connection string
4. Restart GUI to apply changes

### Memory Monitoring

Configure memory thresholds in Settings → Development/Debug:
- **Warning Threshold** - Yellow alert when exceeded
- **Critical Threshold** - Red alert when exceeded
- **Check Interval** - How often to update statistics

## Support

For issues or questions:
1. Check the example configuration file for all available options
2. Review the main Osprey documentation
3. Check the GUI User Manual: `osprey/docs/OSPREY_GUI_USER_MANUAL.md`
4. Contact your facility's Osprey administrator
5. Report bugs via the project's issue tracker