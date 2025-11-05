"""
Agency Master - Voice Calling Agency Orchestrator
Manages voice agents, conversation state, and coordinates LLM interactions.
"""

from typing import List, Dict, Optional, Any
from datetime import datetime
from pydantic import BaseModel
from llm_router import LLMRouter, get_router, Message, LLMRequest, LLMProvider
from enum import Enum


class AgentRole(str, Enum):
    """Available agent roles for voice calling."""
    SALES = "sales"
    SUPPORT = "support"
    APPOINTMENT = "appointment"
    SURVEY = "survey"
    GENERAL = "general"


class ConversationState(str, Enum):
    """Conversation states."""
    INITIALIZED = "initialized"
    ACTIVE = "active"
    WAITING = "waiting"
    COMPLETED = "completed"
    FAILED = "failed"


class ConversationMessage(BaseModel):
    """A message in a conversation."""
    role: str  # 'user', 'assistant', 'system'
    content: str
    timestamp: datetime = datetime.now()


class Conversation(BaseModel):
    """Represents a voice conversation session."""
    conversation_id: str
    agent_role: AgentRole
    state: ConversationState
    messages: List[ConversationMessage] = []
    metadata: Dict[str, Any] = {}
    created_at: datetime = datetime.now()
    updated_at: datetime = datetime.now()


class VoiceAgent:
    """
    Voice agent with specific personality and role.
    Handles conversation flow for voice calling scenarios.
    """

    def __init__(self, role: AgentRole, llm_router: LLMRouter):
        self.role = role
        self.llm_router = llm_router
        self.system_prompts = self._get_system_prompts()

    def _get_system_prompts(self) -> Dict[AgentRole, str]:
        """Get system prompts for different agent roles."""
        return {
            AgentRole.SALES: """You are a professional sales agent making outbound calls.
You are friendly, persuasive, and focused on understanding customer needs.
Keep responses concise and conversational. Ask qualifying questions and guide
the conversation toward a sale or next steps. Be respectful and know when to end the call.""",

            AgentRole.SUPPORT: """You are a helpful customer support agent handling inbound calls.
You are patient, empathetic, and solution-oriented. Listen carefully to customer
issues and provide clear, actionable solutions. Keep responses brief and to the point.
Ask clarifying questions when needed.""",

            AgentRole.APPOINTMENT: """You are an appointment scheduling agent.
Your goal is to schedule appointments efficiently while being courteous and flexible.
Confirm availability, gather necessary information, and provide clear confirmation.
Keep the conversation focused on scheduling.""",

            AgentRole.SURVEY: """You are a survey agent conducting phone surveys.
You are polite, neutral, and focused on collecting accurate information.
Ask questions clearly, one at a time. Thank respondents for their time.
Keep the conversation moving efficiently.""",

            AgentRole.GENERAL: """You are a general-purpose voice assistant.
You are helpful, friendly, and adaptable. Respond naturally to various requests
while keeping responses concise for voice interaction. Be conversational and engaging."""
        }

    def get_system_message(self) -> str:
        """Get the system prompt for this agent's role."""
        return self.system_prompts.get(self.role, self.system_prompts[AgentRole.GENERAL])

    def process_message(
        self,
        user_message: str,
        conversation_history: List[ConversationMessage],
        provider: LLMProvider = LLMProvider.AUTO
    ) -> str:
        """
        Process a user message and generate a response.

        Args:
            user_message: The user's voice input (transcribed)
            conversation_history: Previous messages in the conversation
            provider: Preferred LLM provider

        Returns:
            Agent's response text
        """
        # Build message list for LLM
        messages = [Message(role="system", content=self.get_system_message())]

        # Add conversation history
        for msg in conversation_history[-10:]:  # Keep last 10 messages for context
            messages.append(Message(role=msg.role, content=msg.content))

        # Add current user message
        messages.append(Message(role="user", content=user_message))

        # Call LLM
        request = LLMRequest(
            messages=messages,
            provider=provider,
            temperature=0.8,  # Slightly higher for more natural conversation
            max_tokens=256  # Keep responses concise for voice
        )

        response = self.llm_router.complete(request)
        return response.content


class AgencyMaster:
    """
    Master orchestrator for the voice calling agency.
    Manages multiple conversations and agent assignments.
    """

    def __init__(self):
        self.llm_router = get_router()
        self.conversations: Dict[str, Conversation] = {}
        self.agents: Dict[AgentRole, VoiceAgent] = {}

        # Initialize agents for each role
        for role in AgentRole:
            self.agents[role] = VoiceAgent(role, self.llm_router)

    def create_conversation(
        self,
        conversation_id: str,
        agent_role: AgentRole = AgentRole.GENERAL,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Conversation:
        """
        Create a new conversation session.

        Args:
            conversation_id: Unique identifier for the conversation
            agent_role: Role of the agent handling this conversation
            metadata: Additional metadata about the conversation

        Returns:
            Created Conversation object
        """
        conversation = Conversation(
            conversation_id=conversation_id,
            agent_role=agent_role,
            state=ConversationState.INITIALIZED,
            metadata=metadata or {}
        )

        self.conversations[conversation_id] = conversation
        return conversation

    def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """Get a conversation by ID."""
        return self.conversations.get(conversation_id)

    def process_voice_input(
        self,
        conversation_id: str,
        voice_input: str,
        provider: LLMProvider = LLMProvider.AUTO
    ) -> Dict[str, Any]:
        """
        Process voice input for a conversation.

        Args:
            conversation_id: ID of the conversation
            voice_input: Transcribed voice input from user
            provider: Preferred LLM provider

        Returns:
            Dict containing response and conversation state
        """
        conversation = self.conversations.get(conversation_id)

        if not conversation:
            raise ValueError(f"Conversation {conversation_id} not found")

        # Update conversation state
        if conversation.state == ConversationState.INITIALIZED:
            conversation.state = ConversationState.ACTIVE

        # Add user message to history
        user_message = ConversationMessage(
            role="user",
            content=voice_input,
            timestamp=datetime.now()
        )
        conversation.messages.append(user_message)

        # Get agent for this conversation's role
        agent = self.agents[conversation.agent_role]

        # Process message
        try:
            response_text = agent.process_message(
                voice_input,
                conversation.messages,
                provider
            )

            # Add assistant response to history
            assistant_message = ConversationMessage(
                role="assistant",
                content=response_text,
                timestamp=datetime.now()
            )
            conversation.messages.append(assistant_message)

            # Update conversation
            conversation.updated_at = datetime.now()

            return {
                "conversation_id": conversation_id,
                "response": response_text,
                "state": conversation.state,
                "message_count": len(conversation.messages)
            }

        except Exception as e:
            conversation.state = ConversationState.FAILED
            raise Exception(f"Failed to process voice input: {str(e)}")

    def end_conversation(self, conversation_id: str) -> bool:
        """
        End a conversation session.

        Args:
            conversation_id: ID of the conversation to end

        Returns:
            True if successful
        """
        conversation = self.conversations.get(conversation_id)

        if not conversation:
            return False

        conversation.state = ConversationState.COMPLETED
        conversation.updated_at = datetime.now()
        return True

    def get_conversation_summary(self, conversation_id: str) -> Dict[str, Any]:
        """
        Get a summary of a conversation.

        Args:
            conversation_id: ID of the conversation

        Returns:
            Summary dict with conversation details
        """
        conversation = self.conversations.get(conversation_id)

        if not conversation:
            return {"error": "Conversation not found"}

        return {
            "conversation_id": conversation.conversation_id,
            "agent_role": conversation.agent_role,
            "state": conversation.state,
            "message_count": len(conversation.messages),
            "created_at": conversation.created_at.isoformat(),
            "updated_at": conversation.updated_at.isoformat(),
            "duration_seconds": (conversation.updated_at - conversation.created_at).total_seconds(),
            "metadata": conversation.metadata
        }

    def list_active_conversations(self) -> List[str]:
        """List all active conversation IDs."""
        return [
            conv_id for conv_id, conv in self.conversations.items()
            if conv.state == ConversationState.ACTIVE
        ]

    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of the agency."""
        return {
            "llm_providers": self.llm_router.check_health(),
            "total_conversations": len(self.conversations),
            "active_conversations": len(self.list_active_conversations()),
            "agents_available": list(self.agents.keys())
        }


# Singleton instance
_agency_instance = None


def get_agency() -> AgencyMaster:
    """Get or create singleton AgencyMaster instance."""
    global _agency_instance
    if _agency_instance is None:
        _agency_instance = AgencyMaster()
    return _agency_instance
