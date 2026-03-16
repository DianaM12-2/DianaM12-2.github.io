from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import httpx
import os
from dotenv import load_dotenv

# Optional OpenAI summary (safe fallback if no quota)
try:
    from openai import OpenAI
    import anyio
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False

# Optional Firebase (safe fallback if no service.json)
try:
    import firebase_admin
    from firebase_admin import credentials, firestore
    FIREBASE_AVAILABLE = True
except Exception:
    FIREBASE_AVAILABLE = False


# -------------------
# Config / ENV
# -------------------
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_FACT_CHECK_API_KEY", "").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()

FIREBASE_CRED_PATH = os.getenv("FIREBASE_CRED_PATH", "service.json").strip()

openai_client = OpenAI(api_key=OPENAI_API_KEY) if (OPENAI_AVAILABLE and OPENAI_API_KEY) else None

db = None  # Firestore client if Firebase initializes successfully


def init_firebase():
    global db
    if not FIREBASE_AVAILABLE:
        print("ℹ️ Firebase lib not installed; skipping Firebase.")
        return

    try:
        if os.path.exists(FIREBASE_CRED_PATH):
            # Avoid double-init when uvicorn --reload reloads
            if not firebase_admin._apps:
                cred = credentials.Certificate(FIREBASE_CRED_PATH)
                firebase_admin.initialize_app(cred)
            db = firestore.client()
            print(f"✅ Firebase initialized using: {FIREBASE_CRED_PATH}")
        else:
            print(f"⚠️ Firebase skipped: credential file not found: {FIREBASE_CRED_PATH}")
    except Exception as e:
        print(f"⚠️ Firebase skipped due to error: {e}")
        db = None


init_firebase()


# -------------------
# App
# -------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # dev only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------
# Models
# -------------------
class FactCheckRequest(BaseModel):
    claim: str


class ClaimReview(BaseModel):
    publisher: str
    title: str
    url: str
    text: str
    rating: str


class FactCheckResponse(BaseModel):
    claim: str
    found: bool
    reviews: Optional[List[ClaimReview]]


# -------------------
# Helpers
# -------------------
def normalize_rating(text: str) -> str:
    """Keep rating short + meaningful; prevent long sentences in the UI."""
    if not text:
        return "Unverified"
    t = str(text).strip()

    # If it's insanely long, don't show it as a "rating"
    if len(t) > 40:
        return "Unverified"

    allowed = [
        "true",
        "false",
        "misleading",
        "mostly true",
        "mostly false",
        "incorrect",
        "inaccurate",
        "mixed",
        "unverified",
    ]

    tl = t.lower()
    for kw in allowed:
        if kw in tl:
            # Title-case nicely
            return kw.title()

    return t


def extract_rating(review: Dict[str, Any]) -> str:
    # 1) Structured rating fields
    rating = (
        review.get("reviewRating", {}).get("alternateName")
        or review.get("reviewRating", {}).get("ratingValue")
    )
    if rating:
        return normalize_rating(rating)

    # 2) textualRating
    rating = review.get("textualRating")
    if rating:
        return normalize_rating(rating)

    return "Unverified"


def safe_text(value: Any, fallback: str = "") -> str:
    if value is None:
        return fallback
    s = str(value).strip()
    return s if s else fallback


async def build_ai_summary_review(claim_text: str, source_reviews: List[ClaimReview]) -> ClaimReview:
    """
    Optional AI summary:
    - If OPENAI_API_KEY is missing or quota fails -> return clean fallback.
    - We keep UI professional: no raw error text shown to user.
    """
    if not openai_client:
        return ClaimReview(
            publisher="TrueScope AI",
            title="AI Summary",
            url="#",
            text="AI summary is unavailable right now. Showing fact-check sources below.",
            rating="Unverified",
        )

    # Build a compact prompt from the top few reviews
    top = source_reviews[:6]
    sources_text = "\n".join(
        [f"- {r.publisher}: {r.title} | rating={r.rating} | url={r.url}" for r in top]
    )

    prompt = f"""
You are helping summarize fact-check sources for a prototype app called TrueScope.

Task:
1) Output a one-word label on the first line exactly in this format:
Label: True / False / Misleading / Mixed / Unverified

2) Then provide 2–4 bullet points explaining the consensus across sources.
3) Then provide a short "What to do next" line.

Claim: {claim_text}

Sources:
{sources_text}
""".strip()

    def _call_openai() -> str:
        resp = openai_client.responses.create(
            model=OPENAI_MODEL,
            input=prompt,
        )
        # openai python returns text on output_text
        return resp.output_text.strip()

    try:
        text = await anyio.to_thread.run_sync(_call_openai)
    except Exception as e:
        # Log internally, keep UI clean
        print(f"⚠️ OpenAI summary failed: {type(e).__name__}: {e}")
        return ClaimReview(
            publisher="TrueScope AI",
            title="AI Summary",
            url="#",
            text="AI summary is unavailable right now. Showing fact-check sources below.",
            rating="Unverified",
        )

    # Parse label if present
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    label = "Unverified"
    summary = text

    if lines and lines[0].lower().startswith("label:"):
        label_raw = lines[0].split(":", 1)[1].strip()
        label = normalize_rating(label_raw)
        summary = "\n".join(lines[1:]).strip() or text

    return ClaimReview(
        publisher="TrueScope AI",
        title="AI Summary",
        url="#",
        text=summary,
        rating=label,
    )


# -------------------
# Routes
# -------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "google_key_set": bool(GOOGLE_API_KEY),
        "openai_key_set": bool(OPENAI_API_KEY),
        "firebase_enabled": db is not None,
        "firebase_cred_path": FIREBASE_CRED_PATH,
        "openai_enabled": openai_client is not None,
        "openai_model": OPENAI_MODEL,
    }


@app.post("/fact-check", response_model=FactCheckResponse)
async def fact_check(request: FactCheckRequest):
    claim_text = request.claim.strip()
    if not claim_text:
        raise HTTPException(status_code=400, detail="Claim cannot be empty.")

    if not GOOGLE_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="Missing GOOGLE_FACT_CHECK_API_KEY. Add it to your .env file.",
        )

    url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
    params = {"query": claim_text, "key": GOOGLE_API_KEY}

    async with httpx.AsyncClient(timeout=20) as client:
        resp = await client.get(url, params=params)

    if resp.status_code != 200:
        # Include Google’s error body (useful for debugging)
        try:
            body = resp.json()
        except Exception:
            body = resp.text
        raise HTTPException(
            status_code=500,
            detail=f"Google Fact Check API error (status {resp.status_code}): {body}",
        )

    data = resp.json()
    claims = data.get("claims", [])

    if not claims:
        return FactCheckResponse(claim=claim_text, found=False, reviews=[])

    # Build source reviews (dedupe by url+title)
    seen = set()
    reviews: List[ClaimReview] = []

    for claim in claims:
        for r in claim.get("claimReview", []):
            publisher = safe_text(r.get("publisher", {}).get("name"), "Unknown")
            title = safe_text(r.get("title"), "No Title")
            url_ = safe_text(r.get("url"), "#")
            # Google sometimes provides "text"; sometimes claim text is elsewhere
            text_ = safe_text(r.get("text"), safe_text(claim.get("text"), ""))
            rating = extract_rating(r)

            key = (url_, title)
            if key in seen:
                continue
            seen.add(key)

            reviews.append(
                ClaimReview(
                    publisher=publisher,
                    title=title,
                    url=url_,
                    text=text_ if text_ else "",
                    rating=rating,
                )
            )

    # Optional AI summary inserted first
    ai_review = await build_ai_summary_review(claim_text, reviews)
    reviews = [ai_review] + reviews

    # Save to Firestore if enabled (non-blocking best-effort)
    if db is not None:
        try:
            db.collection("claims").add(
                {
                    "claim": claim_text,
                    "reviews": [rv.model_dump() for rv in reviews],
                    "timestamp": firestore.SERVER_TIMESTAMP,
                }
            )
        except Exception as e:
            print("⚠️ Firestore save failed:", e)

    return FactCheckResponse(claim=claim_text, found=True, reviews=reviews)


# Optional stubs for future work (so UI tabs don’t feel “dead”)
@app.post("/fact-check-image")
async def fact_check_image():
    raise HTTPException(status_code=501, detail="Image fact-check endpoint not implemented yet.")


@app.post("/fact-check-video")
async def fact_check_video():
    raise HTTPException(status_code=501, detail="Video fact-check endpoint not implemented yet.")