import os
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import google.generativeai as genai
from datetime import datetime
from sqlalchemy import create_engine, Column, String, Integer, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Gemini Chatbot API",
    description="API for the Gemini-powered chatbot",
    version="0.1.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database configuration - Using your Render PostgreSQL URL
DATABASE_URL = "postgresql://react_8z5x_user:sZUPOWsl2ChU2DmNGi572dEGdcW7D6DU@dpg-d2cd3madbo4c73bknvkg-a/react_8z5x"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

# Database models
class Conversation(Base):
    __tablename__ = "conversations"
    
    id = Column(String, primary_key=True, index=True)
    title = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class Message(Base):
    __tablename__ = "messages"
    
    id = Column(String, primary_key=True, index=True)
    conversation_id = Column(String, index=True)
    content = Column(Text)
    sender = Column(String)  # 'user' or 'assistant'
    timestamp = Column(DateTime, default=datetime.utcnow)

# Create tables
Base.metadata.create_all(bind=engine)

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Gemini configuration - Using your API key
GEMINI_API_KEY = "AIzaSyBIZbO5KawONRW9-JCBoIQ7vX5EhSKFhNM"
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

# Pydantic models
class MessageRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None

class MessageResponse(BaseModel):
    response: str
    conversation_id: str

class ConversationResponse(BaseModel):
    id: str
    title: str
    created_at: datetime
    updated_at: datetime

class MessageHistoryResponse(BaseModel):
    id: str
    content: str
    sender: str
    timestamp: datetime

# API endpoints
@app.post("/chat", response_model=MessageResponse)
async def chat_with_gemini(
    request: MessageRequest,
    db: Session = Depends(get_db)
):
    try:
        # Start or continue conversation
        conversation_id = request.conversation_id or f"conv-{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}"
        
        # Save user message to database
        user_message = Message(
            id=f"msg-{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}",
            conversation_id=conversation_id,
            content=request.message,
            sender="user",
            timestamp=datetime.utcnow()
        )
        db.add(user_message)
        db.commit()
        
        # Get conversation history for context
        history = db.query(Message).filter(
            Message.conversation_id == conversation_id
        ).order_by(Message.timestamp.asc()).all()
        
        # Format chat history for Gemini
        chat_history = []
        for msg in history:
            role = "user" if msg.sender == "user" else "model"
            chat_history.append({"role": role, "parts": [msg.content]})
        
        # Generate response using Gemini
        response = model.generate_content(
            contents=chat_history,
            generation_config={
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 40
            }
        )
        
        # Extract response text
        response_text = response.text
        
        # Save assistant message to database
        assistant_message = Message(
            id=f"msg-{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}",
            conversation_id=conversation_id,
            content=response_text,
            sender="assistant",
            timestamp=datetime.utcnow()
        )
        db.add(assistant_message)
        
        # Update conversation title if it's the first message
        if not request.conversation_id:
            # Generate a title based on the first message
            title_response = model.generate_content(
                f"Generate a very short (2-4 word) title for a conversation that starts with: {request.message}"
            )
            title = title_response.text.strip('"\'')
            
            conversation = Conversation(
                id=conversation_id,
                title=title,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            db.add(conversation)
        
        db.commit()
        
        return MessageResponse(
            response=response_text,
            conversation_id=conversation_id
        )
        
    except Exception as e:
        logger.error(f"Error in chat_with_gemini: {str(e)}")
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/conversations", response_model=List[ConversationResponse])
async def get_conversations(db: Session = Depends(get_db)):
    conversations = db.query(Conversation).order_by(Conversation.updated_at.desc()).all()
    return conversations

@app.get("/conversations/{conversation_id}/messages", response_model=List[MessageHistoryResponse])
async def get_conversation_messages(
    conversation_id: str,
    db: Session = Depends(get_db)
):
    messages = db.query(Message).filter(
        Message.conversation_id == conversation_id
    ).order_by(Message.timestamp.asc()).all()
    return messages

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
