"""Routing UI Handler for Osprey Framework GUI.

This module handles the display of routing decisions and collection of user feedback
for the multi-project routing system.
"""

from osprey.utils.logger import get_logger

logger = get_logger("routing_ui")


class RoutingUIHandler:
    """Handles routing decision display and user feedback collection."""
    
    def __init__(self, parent_gui):
        """
        Initialize the routing UI handler.
        
        Args:
            parent_gui: Reference to the main OspreyGUI instance
        """
        self.gui = parent_gui
        self.current_routing_decision = None
        self.current_query = None
        self.waiting_for_correction = False
        self.correction_options = []
    
    def display_routing_decision(self, decision):
        """
        Display routing decision to user with visual feedback.
        
        Args:
            decision: RoutingDecision object with routing information
        """
        # Store for feedback collection
        self.current_routing_decision = decision
        
        # Determine color based on confidence
        if decision.confidence >= 0.8:
            confidence_color = "#00FF00"  # Green for high confidence
            confidence_icon = "✅"
        elif decision.confidence >= 0.5:
            confidence_color = "#FFD700"  # Gold for medium confidence
            confidence_icon = "⚠️"
        else:
            confidence_color = "#FFA500"  # Orange for low confidence
            confidence_icon = "⚠️"
        
        # Display routing mode
        if self.gui.router.is_automatic_mode():
            mode_text = "🎯 Automatic Routing"
            mode_color = "#00FFFF"
        else:
            mode_text = "📌 Manual Selection"
            mode_color = "#FFD700"
        
        # Build routing message
        routing_msg = f"\n{mode_text} → {decision.project_name}"
        self.gui._append_colored_message(routing_msg, mode_color)
        
        # Display confidence
        confidence_msg = f"{confidence_icon} Confidence: {decision.confidence:.0%}"
        self.gui._append_colored_message(confidence_msg, confidence_color)
        
        # Display reasoning if available
        if decision.reasoning:
            reasoning_msg = f"   Reason: {decision.reasoning}"
            self.gui._append_colored_message(reasoning_msg, "#808080")
        
        # Display alternatives if available
        if decision.alternative_projects:
            alt_msg = f"   Alternatives: {', '.join(decision.alternative_projects)}"
            self.gui._append_colored_message(alt_msg, "#606060")
        
        # Display feedback prompt (only in automatic mode and if feedback enabled)
        if self.gui.router.is_automatic_mode() and self.gui.settings_manager.get('enable_routing_feedback', True):
            feedback_msg = (
                "   Was this routing correct?\n"
                "   Type 'y' (yes/correct) or 'n' (no/incorrect) to provide feedback\n"
                "   Or type your next query (one query can be queued while processing)"
            )
            self.gui._append_colored_message(feedback_msg, "#87CEEB")
        
        # Add separator
        self.gui._append_colored_message("─" * 60, "#404040")
        
        # Update cache statistics if visible
        if self.gui.show_cache_stats_button.isChecked():
            self.gui._update_cache_statistics()
        
        # Update conversation context if visible
        if self.gui.show_context_button.isChecked():
            self.gui._update_context_display()
    
    def handle_routing_feedback(self, feedback: str, query: str):
        """
        Handle user feedback on routing decision.
        
        Args:
            feedback: User feedback ('y', 'n', 'yes', 'no')
            query: The original query that was routed
        """
        # Check if feedback is enabled
        if not self.gui.settings_manager.get('enable_routing_feedback', True):
            return
        
        if not self.current_routing_decision or not query:
            self.gui._append_colored_message(
                "⚠️ No routing decision to provide feedback for.",
                "#FFA500"
            )
            return
        
        # Store query for correction handling
        self.current_query = query
        
        # Determine if feedback is positive or negative
        is_correct = feedback in ['y', 'yes']
        
        if is_correct:
            # Positive feedback
            self.gui._append_colored_message(
                "✅ Thank you! Routing feedback recorded as correct.",
                "#00FF00"
            )
            
            # Record positive feedback
            self.gui.router.record_routing_feedback(
                query=query,
                selected_project=self.current_routing_decision.project_name,
                confidence=self.current_routing_decision.confidence,
                user_feedback="correct",
                reasoning=self.current_routing_decision.reasoning
            )
            
            # Clear current routing decision (user answered feedback)
            self.current_routing_decision = None
            self.current_query = None
            
        else:
            # Negative feedback - ask for correct project
            self.gui._append_colored_message(
                "👎 Routing was incorrect. Which project should have been used?",
                "#FFA500"
            )
            
            # Get enabled projects for selection
            enabled_projects = self.gui.project_manager.get_enabled_projects()
            project_names = [p.metadata.name for p in enabled_projects]
            
            # Display options
            options_msg = "Available projects:\n" + "\n".join(
                f"  {i+1}. {name}" for i, name in enumerate(project_names)
            )
            self.gui._append_colored_message(options_msg, "#FFFFFF")
            self.gui._append_colored_message(
                "Enter the number or name of the correct project:",
                "#87CEEB"
            )
            
            # Set state to wait for correction
            self.waiting_for_correction = True
            self.correction_options = project_names
            return
        
        # Clear current routing decision
        self.current_routing_decision = None
        self.current_query = None
    
    def handle_correction_input(self, user_input: str):
        """
        Handle user input for routing correction.
        
        Args:
            user_input: User's correction input (project name or number)
        """
        # Check if feedback is enabled
        if not self.gui.settings_manager.get('enable_routing_feedback', True):
            self.waiting_for_correction = False
            return
        
        if not self.correction_options:
            self.gui._append_colored_message(
                "⚠️ No correction options available.",
                "#FFA500"
            )
            self.waiting_for_correction = False
            return
        
        # Try to parse as number
        correct_project = None
        try:
            index = int(user_input) - 1
            if 0 <= index < len(self.correction_options):
                correct_project = self.correction_options[index]
        except ValueError:
            # Not a number, try as project name
            if user_input in self.correction_options:
                correct_project = user_input
        
        if not correct_project:
            self.gui._append_colored_message(
                f"⚠️ Invalid selection: '{user_input}'. Please try again.",
                "#FFA500"
            )
            return
        
        # Record negative feedback with correction
        self.gui.router.record_routing_feedback(
            query=self.current_query,
            selected_project=self.current_routing_decision.project_name,
            confidence=self.current_routing_decision.confidence,
            user_feedback="incorrect",
            correct_project=correct_project,
            reasoning=self.current_routing_decision.reasoning
        )
        
        self.gui._append_colored_message(
            f"✅ Thank you! Feedback recorded. Correct project: {correct_project}",
            "#00FF00"
        )
        self.gui._append_colored_message(
            "   The system will learn from this correction.",
            "#87CEEB"
        )
        
        # Clear state
        self.current_routing_decision = None
        self.current_query = None
        self.waiting_for_correction = False
        self.correction_options = []
    
    def is_waiting_for_correction(self) -> bool:
        """Check if handler is waiting for correction input."""
        return self.waiting_for_correction
    
    def has_pending_feedback(self) -> bool:
        """Check if there's a pending routing decision for feedback."""
        return self.current_routing_decision is not None
    
    def clear_feedback_state(self):
        """Clear all feedback-related state."""
        self.current_routing_decision = None
        self.current_query = None
        self.waiting_for_correction = False
        self.correction_options = []