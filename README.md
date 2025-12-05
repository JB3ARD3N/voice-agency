# 🔮 0RB VOICE AGENCY - Deployment Guide

## Quick Start (Local Testing)

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate  # or: venv\Scripts\activate on Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Copy env file
cp .env.example .env
# Edit .env with your credentials

# 4. Run server
python server.py
```

Visit: http://localhost:8000/dashboard

---

## Deploy to Railway (Recommended)

### 1. Push to GitHub
```bash
git init
git add .
git commit -m "Initial voice agency deploy"
git remote add origin https://github.com/YOUR_USERNAME/voice-agency.git
git push -u origin main
```

### 2. Deploy on Railway
1. Go to https://railway.app
2. New Project → Deploy from GitHub repo
3. Add environment variables (from .env.example)
4. Deploy!

### 3. Configure Twilio
1. Go to Twilio Console
2. Buy a phone number (or use existing)
3. Set webhook URL: `https://YOUR-RAILWAY-URL.up.railway.app/voice/incoming`
4. Method: POST

---

## Deploy to Fly.io

```bash
# Install flyctl
curl -L https://fly.io/install.sh | sh

# Login and deploy
fly auth login
fly launch
fly secrets set TWILIO_ACCOUNT_SID=xxx TWILIO_AUTH_TOKEN=xxx OPENAI_API_KEY=xxx
fly deploy
```

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/dashboard` | GET | Web dashboard |
| `/voice/incoming` | POST | Twilio webhook (incoming calls) |
| `/voice/respond` | POST | Twilio webhook (speech responses) |
| `/business/register` | POST | Register a business |
| `/leads` | GET | Get captured leads |
| `/test/call` | POST | Test AI response |

---

## Business Registration Example

```bash
curl -X POST http://localhost:8000/business/register \
  -H "Content-Type: application/json" \
  -d '{
    "id": "acme-hvac",
    "name": "ACME HVAC Services",
    "phone": "+15551234567",
    "industry": "hvac",
    "greeting": "Thank you for calling ACME HVAC! How can we help you today?",
    "services": ["AC Repair", "Heating", "Installation", "Maintenance"],
    "hours": {"Mon-Fri": "7am-7pm", "Sat": "8am-4pm", "Sun": "Emergency only"}
  }'
```

---

## Revenue Model

| Tier | Price | Includes |
|------|-------|----------|
| Starter | $500/mo | 500 minutes, lead capture, 1 business |
| Growth | $1,000/mo | 2,000 minutes, analytics, 3 businesses |
| Enterprise | $2,000/mo | Unlimited, custom voice, white-label |

+ 10% of booked revenue from qualified leads

---

## Next Steps

1. ✅ Deploy server
2. ⬜ Configure Twilio webhook
3. ⬜ Register first business
4. ⬜ Test with real call
5. ⬜ Get first paying client

---

Built by 0RB Empire | Love • Loyalty • Honor | Everybody Eats
