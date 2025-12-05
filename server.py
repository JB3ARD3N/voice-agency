#!/usr/bin/env python3
"""
0RB VOICE AGENCY - Production Server
=====================================
AI-powered voice calling for local businesses
Handles: Appointments, Lead Qualification, Customer Service

Stack:
- FastAPI (async, production-grade)
- Twilio (voice calls)
- Deepgram (STT - 100ms)
- OpenAI/Ollama (intent + response)
- ElevenLabs/Deepgram (TTS)

Revenue Model: $500/mo base + 10% of booked revenue
"""

import os
import json
import asyncio
from datetime import datetime
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, Request, HTTPException, WebSocket
from fastapi.responses import Response, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════

class Config:
    # Twilio
    TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
    TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")
    TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER", "")
    
    # AI Providers (free tier first)
    OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "")
    ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
    
    # Server
    BASE_URL = os.getenv("BASE_URL", "https://voice.0r8.ai")
    
    # Business defaults
    DEFAULT_BUSINESS_NAME = "Our Business"
    DEFAULT_GREETING = "Hello, thank you for calling. How can I help you today?"

config = Config()

# ═══════════════════════════════════════════════════════════════
# MODELS
# ═══════════════════════════════════════════════════════════════

class Business(BaseModel):
    id: str
    name: str
    phone: str
    industry: str  # hvac, plumbing, dental, etc.
    greeting: str
    services: List[str]
    hours: Dict[str, str]
    booking_url: Optional[str] = None
    
class CallSession(BaseModel):
    call_sid: str
    business_id: str
    caller_phone: str
    started_at: datetime
    transcript: List[Dict[str, str]] = []
    intent: Optional[str] = None
    outcome: Optional[str] = None
    booked: bool = False

class LeadData(BaseModel):
    name: Optional[str] = None
    phone: str
    email: Optional[str] = None
    service_needed: Optional[str] = None
    preferred_time: Optional[str] = None
    notes: str = ""
    qualified: bool = False
    score: int = 0  # 1-10

# ═══════════════════════════════════════════════════════════════
# IN-MEMORY STORAGE (Replace with DB in production)
# ═══════════════════════════════════════════════════════════════

businesses: Dict[str, Business] = {}
active_calls: Dict[str, CallSession] = {}
leads: List[LeadData] = []

# ═══════════════════════════════════════════════════════════════
# AI ROUTER - Free tier first, fallback to paid
# ═══════════════════════════════════════════════════════════════

class AIRouter:
    """Smart routing: Ollama (free) → OpenAI (paid)"""
    
    def __init__(self):
        self.ollama_available = False
        self.check_ollama()
    
    def check_ollama(self):
        """Check if Ollama is running"""
        try:
            import requests
            r = requests.get(f"{config.OLLAMA_URL}/api/tags", timeout=2)
            self.ollama_available = r.status_code == 200
        except:
            self.ollama_available = False
    
    async def generate(self, prompt: str, system: str = "") -> str:
        """Generate response using best available model"""
        
        # Try Ollama first (FREE)
        if self.ollama_available:
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{config.OLLAMA_URL}/api/generate",
                        json={
                            "model": "llama3",
                            "prompt": prompt,
                            "system": system,
                            "stream": False
                        },
                        timeout=30.0
                    )
                    if response.status_code == 200:
                        return response.json().get("response", "")
            except Exception as e:
                print(f"Ollama failed: {e}")
        
        # Fallback to OpenAI
        if config.OPENAI_API_KEY:
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        "https://api.openai.com/v1/chat/completions",
                        headers={"Authorization": f"Bearer {config.OPENAI_API_KEY}"},
                        json={
                            "model": "gpt-4o-mini",
                            "messages": [
                                {"role": "system", "content": system},
                                {"role": "user", "content": prompt}
                            ],
                            "max_tokens": 150
                        },
                        timeout=30.0
                    )
                    if response.status_code == 200:
                        return response.json()["choices"][0]["message"]["content"]
            except Exception as e:
                print(f"OpenAI failed: {e}")
        
        # Last resort
        return "I apologize, I'm having trouble processing that. Could you please repeat?"

ai_router = AIRouter()

# ═══════════════════════════════════════════════════════════════
# INTENT DETECTION
# ═══════════════════════════════════════════════════════════════

class IntentEngine:
    """Detect caller intent from speech"""
    
    INTENTS = {
        "appointment": ["book", "schedule", "appointment", "available", "opening", "slot", "come in"],
        "pricing": ["cost", "price", "how much", "rate", "fee", "charge", "quote"],
        "hours": ["open", "close", "hours", "when", "time"],
        "emergency": ["emergency", "urgent", "asap", "right now", "immediately", "broken", "leak", "not working"],
        "info": ["tell me about", "what do you", "services", "do you offer"],
        "human": ["speak to", "talk to", "real person", "representative", "manager", "human"]
    }
    
    def detect(self, text: str) -> tuple[str, float]:
        """Returns (intent, confidence)"""
        text_lower = text.lower()
        
        scores = {}
        for intent, keywords in self.INTENTS.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > 0:
                scores[intent] = score
        
        if not scores:
            return ("general", 0.5)
        
        best_intent = max(scores, key=scores.get)
        confidence = min(scores[best_intent] / 3, 1.0)
        
        return (best_intent, confidence)

intent_engine = IntentEngine()

# ═══════════════════════════════════════════════════════════════
# VOICE AGENT - The actual conversation handler
# ═══════════════════════════════════════════════════════════════

class VoiceAgent:
    """Handles voice conversations for businesses"""
    
    def __init__(self, business: Business):
        self.business = business
        self.system_prompt = self._build_system_prompt()
    
    def _build_system_prompt(self) -> str:
        return f"""You are a friendly, professional AI receptionist for {self.business.name}.

BUSINESS INFO:
- Industry: {self.business.industry}
- Services: {', '.join(self.business.services)}
- Hours: {json.dumps(self.business.hours)}

YOUR ROLE:
1. Answer questions about the business
2. Book appointments when requested
3. Capture lead information (name, phone, service needed)
4. Transfer to human for complex issues

RULES:
- Be warm, professional, concise
- Keep responses under 2 sentences when possible
- Always confirm details before ending
- If unsure, offer to have someone call back
- Never make up information

RESPONSE FORMAT:
- Speak naturally, as if on a phone call
- No bullet points or formatting
- Use conversational language"""
    
    async def respond(self, caller_input: str, context: List[Dict] = None) -> str:
        """Generate response to caller"""
        
        # Detect intent
        intent, confidence = intent_engine.detect(caller_input)
        
        # Build context
        history = ""
        if context:
            history = "\n".join([f"{m['role']}: {m['content']}" for m in context[-6:]])
        
        prompt = f"""Previous conversation:
{history}

Caller just said: "{caller_input}"
Detected intent: {intent} (confidence: {confidence:.0%})

Respond appropriately as the receptionist. Keep it brief and natural."""
        
        response = await ai_router.generate(prompt, self.system_prompt)
        return response.strip()
    
    async def extract_lead_info(self, transcript: List[Dict]) -> LeadData:
        """Extract lead information from conversation"""
        
        full_text = " ".join([m["content"] for m in transcript if m["role"] == "caller"])
        
        prompt = f"""Extract lead information from this conversation transcript:

{full_text}

Return JSON with these fields (use null if not mentioned):
- name: caller's name
- service_needed: what service they want
- preferred_time: when they want to come in
- notes: any other important details
- qualified: true if they seem like a real lead (not spam/wrong number)
- score: 1-10 how likely to convert"""
        
        response = await ai_router.generate(prompt, "You are a data extraction assistant. Return only valid JSON.")
        
        try:
            data = json.loads(response)
            return LeadData(**data, phone="unknown")
        except:
            return LeadData(phone="unknown", notes=full_text[:500])

# ═══════════════════════════════════════════════════════════════
# FASTAPI APPLICATION
# ═══════════════════════════════════════════════════════════════

app = FastAPI(
    title="0RB Voice Agency",
    description="AI-powered voice calling for local businesses",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ═══════════════════════════════════════════════════════════════
# TWILIO WEBHOOKS
# ═══════════════════════════════════════════════════════════════

@app.post("/voice/incoming")
async def handle_incoming_call(request: Request):
    """Handle incoming Twilio call"""
    form = await request.form()
    
    call_sid = form.get("CallSid", "")
    caller = form.get("From", "")
    called = form.get("To", "")
    
    # Find business by phone number
    business = None
    for b in businesses.values():
        if b.phone == called:
            business = b
            break
    
    if not business:
        # Default response
        twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="Polly.Joanna">Thank you for calling. We are currently unavailable. Please leave a message after the beep.</Say>
    <Record maxLength="120" transcribe="true"/>
</Response>"""
        return Response(content=twiml, media_type="application/xml")
    
    # Create call session
    session = CallSession(
        call_sid=call_sid,
        business_id=business.id,
        caller_phone=caller,
        started_at=datetime.now()
    )
    active_calls[call_sid] = session
    
    # Return greeting with gather
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="Polly.Joanna">{business.greeting}</Say>
    <Gather input="speech" timeout="5" speechTimeout="auto" action="/voice/respond?call_sid={call_sid}" method="POST">
        <Say voice="Polly.Joanna">How can I help you?</Say>
    </Gather>
    <Say voice="Polly.Joanna">I didn't catch that. Please call back if you need assistance.</Say>
</Response>"""
    
    return Response(content=twiml, media_type="application/xml")

@app.post("/voice/respond")
async def handle_voice_response(request: Request, call_sid: str):
    """Handle caller speech and respond"""
    form = await request.form()
    
    speech_result = form.get("SpeechResult", "")
    
    session = active_calls.get(call_sid)
    if not session:
        return Response(content="<Response><Hangup/></Response>", media_type="application/xml")
    
    business = businesses.get(session.business_id)
    if not business:
        return Response(content="<Response><Hangup/></Response>", media_type="application/xml")
    
    # Log caller input
    session.transcript.append({"role": "caller", "content": speech_result})
    
    # Generate AI response
    agent = VoiceAgent(business)
    response_text = await agent.respond(speech_result, session.transcript)
    
    # Log AI response
    session.transcript.append({"role": "agent", "content": response_text})
    
    # Check for end conditions
    intent, _ = intent_engine.detect(speech_result)
    
    if intent == "human":
        # Transfer to human
        twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="Polly.Joanna">Let me transfer you to a team member. Please hold.</Say>
    <Dial>{business.phone}</Dial>
</Response>"""
    elif "goodbye" in speech_result.lower() or "thank you" in speech_result.lower():
        # End call
        twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="Polly.Joanna">{response_text} Thank you for calling {business.name}. Have a great day!</Say>
    <Hangup/>
</Response>"""
        # Extract lead info
        asyncio.create_task(process_completed_call(session, agent))
    else:
        # Continue conversation
        twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="Polly.Joanna">{response_text}</Say>
    <Gather input="speech" timeout="5" speechTimeout="auto" action="/voice/respond?call_sid={call_sid}" method="POST"/>
    <Say voice="Polly.Joanna">Are you still there?</Say>
    <Gather input="speech" timeout="3" speechTimeout="auto" action="/voice/respond?call_sid={call_sid}" method="POST"/>
</Response>"""
    
    return Response(content=twiml, media_type="application/xml")

async def process_completed_call(session: CallSession, agent: VoiceAgent):
    """Process completed call - extract lead, store data"""
    lead = await agent.extract_lead_info(session.transcript)
    lead.phone = session.caller_phone
    leads.append(lead)
    
    # Cleanup
    if session.call_sid in active_calls:
        del active_calls[session.call_sid]

# ═══════════════════════════════════════════════════════════════
# API ENDPOINTS
# ═══════════════════════════════════════════════════════════════

@app.get("/")
async def root():
    """Health check"""
    return {
        "status": "online",
        "service": "0RB Voice Agency",
        "version": "1.0.0",
        "businesses": len(businesses),
        "active_calls": len(active_calls),
        "leads_captured": len(leads)
    }

@app.post("/business/register")
async def register_business(business: Business):
    """Register a new business"""
    businesses[business.id] = business
    return {"status": "registered", "business_id": business.id}

@app.get("/business/{business_id}")
async def get_business(business_id: str):
    """Get business details"""
    if business_id not in businesses:
        raise HTTPException(404, "Business not found")
    return businesses[business_id]

@app.get("/leads")
async def get_leads(business_id: Optional[str] = None, qualified_only: bool = False):
    """Get captured leads"""
    result = leads
    if qualified_only:
        result = [l for l in result if l.qualified]
    return {"leads": result, "count": len(result)}

@app.get("/calls/active")
async def get_active_calls():
    """Get active calls"""
    return {"calls": list(active_calls.values()), "count": len(active_calls)}

@app.post("/test/call")
async def test_call(text: str, business_id: str = "demo"):
    """Test the AI response without making a real call"""
    business = businesses.get(business_id)
    if not business:
        # Create demo business
        business = Business(
            id="demo",
            name="Demo HVAC Company",
            phone="+1234567890",
            industry="hvac",
            greeting="Thank you for calling Demo HVAC!",
            services=["AC repair", "Heating", "Installation", "Maintenance"],
            hours={"Mon-Fri": "8am-6pm", "Sat": "9am-2pm", "Sun": "Closed"}
        )
        businesses["demo"] = business
    
    agent = VoiceAgent(business)
    response = await agent.respond(text)
    intent, confidence = intent_engine.detect(text)
    
    return {
        "input": text,
        "response": response,
        "intent": intent,
        "confidence": confidence
    }

# ═══════════════════════════════════════════════════════════════
# DASHBOARD
# ═══════════════════════════════════════════════════════════════

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    """Simple dashboard"""
    return """
<!DOCTYPE html>
<html>
<head>
    <title>0RB Voice Agency</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: -apple-system, BlinkMacSystemFont, sans-serif;
            background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 100%);
            color: #fff;
            min-height: 100vh;
            padding: 2rem;
        }
        .container { max-width: 1200px; margin: 0 auto; }
        h1 { 
            font-size: 2.5rem; 
            margin-bottom: 2rem;
            background: linear-gradient(90deg, #00f5d4, #7b2cbf);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1.5rem;
            margin-bottom: 2rem;
        }
        .stat-card {
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 12px;
            padding: 1.5rem;
        }
        .stat-value { 
            font-size: 2.5rem; 
            font-weight: bold;
            color: #00f5d4;
        }
        .stat-label { 
            color: rgba(255,255,255,0.6);
            margin-top: 0.5rem;
        }
        .test-section {
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 12px;
            padding: 2rem;
        }
        input, button {
            padding: 1rem;
            border-radius: 8px;
            border: 1px solid rgba(255,255,255,0.2);
            background: rgba(255,255,255,0.1);
            color: #fff;
            font-size: 1rem;
        }
        input { width: 70%; margin-right: 1rem; }
        button { 
            background: linear-gradient(90deg, #00f5d4, #7b2cbf);
            border: none;
            cursor: pointer;
            font-weight: bold;
        }
        button:hover { opacity: 0.9; }
        #response {
            margin-top: 1.5rem;
            padding: 1rem;
            background: rgba(0,0,0,0.3);
            border-radius: 8px;
            white-space: pre-wrap;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔮 0RB Voice Agency</h1>
        
        <div class="stats" id="stats">
            <div class="stat-card">
                <div class="stat-value" id="businesses">-</div>
                <div class="stat-label">Businesses</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="active">-</div>
                <div class="stat-label">Active Calls</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="leads">-</div>
                <div class="stat-label">Leads Captured</div>
            </div>
        </div>
        
        <div class="test-section">
            <h2 style="margin-bottom: 1rem;">Test AI Response</h2>
            <div>
                <input type="text" id="testInput" placeholder="Type what a caller would say..." />
                <button onclick="testCall()">Test</button>
            </div>
            <div id="response"></div>
        </div>
    </div>
    
    <script>
        async function loadStats() {
            const res = await fetch('/');
            const data = await res.json();
            document.getElementById('businesses').textContent = data.businesses;
            document.getElementById('active').textContent = data.active_calls;
            document.getElementById('leads').textContent = data.leads_captured;
        }
        
        async function testCall() {
            const input = document.getElementById('testInput').value;
            const res = await fetch('/test/call?text=' + encodeURIComponent(input) + '&business_id=demo', {
                method: 'POST'
            });
            const data = await res.json();
            document.getElementById('response').textContent = 
                'Intent: ' + data.intent + ' (' + (data.confidence * 100).toFixed(0) + '%)\\n\\n' +
                'AI Response:\\n' + data.response;
        }
        
        loadStats();
        setInterval(loadStats, 5000);
    </script>
</body>
</html>
"""

# ═══════════════════════════════════════════════════════════════
# STARTUP
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    
    # Create demo business on startup
    demo = Business(
        id="demo",
        name="Demo HVAC Company",
        phone="+1234567890",
        industry="hvac",
        greeting="Thank you for calling Demo HVAC, your comfort is our priority!",
        services=["AC repair", "Heating", "Installation", "Maintenance", "Emergency service"],
        hours={"Mon-Fri": "8am-6pm", "Sat": "9am-2pm", "Sun": "Emergency only"}
    )
    businesses["demo"] = demo
    
    print("🔮 0RB Voice Agency Starting...")
    print(f"   Ollama: {'✓ Connected' if ai_router.ollama_available else '✗ Not available'}")
    print(f"   OpenAI: {'✓ Configured' if config.OPENAI_API_KEY else '✗ Not configured'}")
    print(f"   Twilio: {'✓ Configured' if config.TWILIO_ACCOUNT_SID else '✗ Not configured'}")
    print("")
    print("   Dashboard: http://localhost:8000/dashboard")
    print("   API Docs:  http://localhost:8000/docs")
    print("")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
