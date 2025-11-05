"""
Voice Agency API Server
FastAPI server providing REST endpoints for voice calling agency interactions.
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
import uuid

from agency_master import (
    get_agency,
    AgentRole,
    ConversationState
)
from llm_router import LLMProvider

# Initialize FastAPI app
app = FastAPI(
    title="Voice Calling Agency API",
    description="API for managing voice-based AI agent conversations",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize agency
agency = get_agency()


# Request/Response Models
class CreateConversationRequest(BaseModel):
    """Request to create a new conversation."""
    agent_role: AgentRole = AgentRole.GENERAL
    metadata: Optional[Dict[str, Any]] = {}


class CreateConversationResponse(BaseModel):
    """Response for conversation creation."""
    conversation_id: str
    agent_role: str
    state: str
    created_at: str


class VoiceInputRequest(BaseModel):
    """Request to process voice input."""
    voice_input: str = Field(..., description="Transcribed voice input from user")
    provider: Optional[LLMProvider] = LLMProvider.AUTO


class VoiceInputResponse(BaseModel):
    """Response for voice input processing."""
    conversation_id: str
    response: str
    state: str
    message_count: int


class ConversationSummaryResponse(BaseModel):
    """Response for conversation summary."""
    conversation_id: str
    agent_role: str
    state: str
    message_count: int
    created_at: str
    updated_at: str
    duration_seconds: float
    metadata: Dict[str, Any]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: str
    agency_status: Dict[str, Any]


class ErrorResponse(BaseModel):
    """Error response."""
    error: str
    detail: Optional[str] = None


# API Endpoints

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "service": "Voice Calling Agency API",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "health": "/health",
            "conversations": "/conversations",
            "voice_input": "/conversations/{conversation_id}/voice",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint.
    Returns status of the agency and LLM providers.
    """
    agency_status = agency.get_health_status()

    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        agency_status=agency_status
    )


@app.post(
    "/conversations",
    response_model=CreateConversationResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Conversations"]
)
async def create_conversation(request: CreateConversationRequest):
    """
    Create a new conversation session.

    - **agent_role**: The role of the agent (sales, support, appointment, survey, general)
    - **metadata**: Optional metadata about the conversation
    """
    try:
        # Generate unique conversation ID
        conversation_id = str(uuid.uuid4())

        # Create conversation
        conversation = agency.create_conversation(
            conversation_id=conversation_id,
            agent_role=request.agent_role,
            metadata=request.metadata
        )

        return CreateConversationResponse(
            conversation_id=conversation.conversation_id,
            agent_role=conversation.agent_role,
            state=conversation.state,
            created_at=conversation.created_at.isoformat()
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create conversation: {str(e)}"
        )


@app.post(
    "/conversations/{conversation_id}/voice",
    response_model=VoiceInputResponse,
    tags=["Conversations"]
)
async def process_voice_input(
    conversation_id: str,
    request: VoiceInputRequest
):
    """
    Process voice input for a conversation.

    - **conversation_id**: ID of the conversation
    - **voice_input**: Transcribed voice input from the user
    - **provider**: Optional LLM provider preference (auto, xai, groq)
    """
    try:
        result = agency.process_voice_input(
            conversation_id=conversation_id,
            voice_input=request.voice_input,
            provider=request.provider
        )

        return VoiceInputResponse(**result)

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process voice input: {str(e)}"
        )


@app.get(
    "/conversations/{conversation_id}",
    response_model=ConversationSummaryResponse,
    tags=["Conversations"]
)
async def get_conversation(conversation_id: str):
    """
    Get conversation details and summary.

    - **conversation_id**: ID of the conversation
    """
    summary = agency.get_conversation_summary(conversation_id)

    if "error" in summary:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=summary["error"]
        )

    return ConversationSummaryResponse(**summary)


@app.post(
    "/conversations/{conversation_id}/end",
    tags=["Conversations"]
)
async def end_conversation(conversation_id: str):
    """
    End a conversation session.

    - **conversation_id**: ID of the conversation to end
    """
    success = agency.end_conversation(conversation_id)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation {conversation_id} not found"
        )

    return {
        "conversation_id": conversation_id,
        "status": "ended",
        "timestamp": datetime.now().isoformat()
    }


@app.get(
    "/conversations",
    response_model=List[str],
    tags=["Conversations"]
)
async def list_active_conversations():
    """
    List all active conversation IDs.
    """
    return agency.list_active_conversations()


@app.get(
    "/agents",
    tags=["Agents"]
)
async def list_agents():
    """
    List available agent roles.
    """
    return {
        "agents": [
            {
                "role": "sales",
                "description": "Professional sales agent for outbound calls"
            },
            {
                "role": "support",
                "description": "Customer support agent for inbound calls"
            },
            {
                "role": "appointment",
                "description": "Appointment scheduling agent"
            },
            {
                "role": "survey",
                "description": "Survey agent for conducting phone surveys"
            },
            {
                "role": "general",
                "description": "General-purpose voice assistant"
            }
        ]
    }


# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Handle 404 errors."""
    return {
        "error": "Not Found",
        "detail": "The requested resource was not found"
    }


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """Handle 500 errors."""
    return {
        "error": "Internal Server Error",
        "detail": "An unexpected error occurred"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
