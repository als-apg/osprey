# GUI Conversation History Preprocessing

## Problem

When users ask questions like "Do you know what the weather was like in prague when i asked earlier?", the orchestrator was incorrectly planning to use the `time_range_parsing` capability to extract the timestamp from conversation history. However, `time_range_parsing` can only parse time expressions from the task objective string itself - it cannot access conversation history.

This caused the error:
```
AmbiguousTimeReferenceError: No time range found in query: 'Parse the timestamp of the user's earlier question to create a time range representing the moment the question was asked.'
```

## Solution

Since only the GUI has access to conversation history (not the CLI), we implemented a **GUI-specific preprocessing step** that resolves conversation history references before sending messages to the framework.

## Implementation

### 1. ConversationPreprocessor (`osprey/src/osprey/interfaces/pyqt/conversation_preprocessor.py`)

A new preprocessor class that:
- Detects messages containing conversation history references (e.g., "earlier", "when I asked before")
- Looks up the actual timestamp from conversation history
- Enhances the message with explicit timestamp information

Example transformation:
```
Original:  "Do you know what the weather was like in prague when i asked earlier?"
Enhanced:  "Do you know what the weather was like in prague at 2026-01-04 10:30:00 when I asked earlier?"
```

### 2. Integration in GUI (`gui.py`)

The `send_message()` method now:
1. Preprocesses the user message to resolve history references
2. Shows an explanation to the user about what was resolved
3. Sends the enhanced message to the framework

This happens transparently - the framework receives a message with explicit timestamps that `time_range_parsing` can handle.

## Benefits

1. **GUI-Only Feature**: Leverages GUI's conversation history without requiring framework changes
2. **Transparent**: Users see what was resolved via the explanation message
3. **Framework Compatible**: Enhanced messages work with existing framework capabilities
4. **No CLI Impact**: CLI behavior unchanged (it doesn't have conversation history anyway)

## Example Usage

**User Input:**
```
Do you know what the weather was like in prague when i asked earlier?
```

**GUI Shows:**
```
📝 Resolved conversation reference:
   Previous question: "What's the weather in Prague?"
   Asked at: 2026-01-04 10:30:00
```

**Framework Receives:**
```
Do you know what the weather was like in prague at 2026-01-04 10:30:00 when I asked earlier?
```

**Result:**
The `time_range_parsing` capability can now extract "2026-01-04 10:30:00" from the task objective string, and the weather capability can retrieve weather data for that specific time.

## Testing

Run the test to verify preprocessing works:
```bash
cd osprey
python -m pytest tests/gui/test_conversation_preprocessor.py -v
```

Or run directly:
```bash
python tests/gui/test_conversation_preprocessor.py
```

## Future Enhancements

Potential improvements:
- Support for multiple previous messages ("the question before that")
- Smarter detection of which previous message is being referenced
- Support for relative time expressions ("2 hours ago")
- Caching of preprocessed messages to avoid redundant processing