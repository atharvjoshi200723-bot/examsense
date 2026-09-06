# ⚡ ExamSense
### AI-Powered Exam Question Predictor for Indian Private Universities

> Predict your exam questions using 10+ years of past papers — built with RAG (Retrieval-Augmented Generation)

---

## 🏫 Supported Colleges
- VIT Vellore
- SRM Institute of Science and Technology
- BITS Pilani
- Graphic Era Hill University
- UPES Dehradun
- Manipal Academy of Higher Education

## ✨ Features
- 🔥 Top 15 Most Likely Questions with confidence scores
- 📄 AI Generated Model Exam Paper
- 🃏 Flashcards from past papers
- 🧠 MCQ Quiz Mode
- 💬 Ask Anything from your syllabus
- 📤 Upload papers to earn free Pro access

## 🛠 Tech Stack
- **Backend**: FastAPI (Python)
- **Frontend**: Next.js + Tailwind CSS
- **AI**: Google Gemini API (text-embedding-004 + Gemini 1.5 Flash)
- **Vector DB**: ChromaDB
- **Hosting**: Render (backend) + Vercel (frontend)
- **Payments**: Razorpay

## 🚀 Getting Started

### Backend
```bash
cd backend
pip install -r requirements.txt
cp .env.example .env
# Add your GEMINI_API_KEY to .env
python ingest_all.py   # Download and process all papers (run once)
uvicorn api.main:app --reload
```

### Frontend
```bash
cd frontend
npm install
cp .env.example .env.local
# Add your backend URL to .env.local
npm run dev
```

## 📁 Project Structure
```
examSense/
├── backend/
│   ├── scrapers/        ← Download papers from GitHub + portals
│   ├── processors/      ← PDF reading, RAG, predictions
│   ├── api/             ← FastAPI server
│   └── data/            ← Downloaded papers + ChromaDB storage
├── frontend/
│   ├── pages/           ← Next.js pages
│   └── components/      ← Reusable UI components
└── README.md
```

## 💰 Pricing
| Plan | Price |
|------|-------|
| Free | ₹0 — Top 3 predictions, basic features |
| Pro Monthly | ₹99/month — Full access |
| Pro Semester | ₹199/semester — Best value |

---
Built with ❤️ for Indian college students
