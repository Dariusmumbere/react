import os
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import google.generativeai as genai
from datetime import datetime
from sqlalchemy import create_engine, Column, String, Integer, DateTime, Text, desc
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import logging
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Gemini Chatbot API",
    description="API for the Gemini-powered chatbot with conversation history",
    version="0.2.0",
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

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://react_8z5x_user:sZUPOWsl2ChU2DmNGi572dEGdcW7D6DU@dpg-d2cd3madbo4c73bknvkg-a/react_8z5x")
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

# Gemini configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyBIZbO5KawONRW9-JCBoIQ7vX5EhSKFhNM")
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-2.5-flash')

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
    preview: Optional[str] = None  # Last message preview

class MessageHistoryResponse(BaseModel):
    id: str
    content: str
    sender: str
    timestamp: datetime

# Helper functions
def generate_conversation_id():
    return f"conv-{uuid.uuid4()}"

def generate_message_id():
    return f"msg-{uuid.uuid4()}"

def generate_conversation_title(first_message: str, model) -> str:
    try:
        prompt = f"""Generate a very short (2-4 word) title for a conversation that starts with: 
        "{first_message}". Return only the title, no quotes or additional text."""
        response = model.generate_content(prompt)
        return response.text.strip('"\'')
    except Exception as e:
        logger.warning(f"Failed to generate title: {e}")
        return first_message[:30] + "..." if len(first_message) > 30 else first_message

# API endpoints
@app.post("/chat", response_model=MessageResponse)
async def chat_with_gemini(
    request: MessageRequest,
    db: Session = Depends(get_db)
):
    try:
        # Start or continue conversation
        conversation_id = request.conversation_id or generate_conversation_id()
        
        # Save user message to database
        user_message = Message(
            id=generate_message_id(),
            conversation_id=conversation_id,
            content=request.message,
            sender="user",
            timestamp=datetime.utcnow()
        )
        db.add(user_message)
        db.commit()
        
        # Get conversation history for context (last 10 messages)
        history = db.query(Message).filter(
            Message.conversation_id == conversation_id
        ).order_by(Message.timestamp.asc()).limit(10).all()
        
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
            id=generate_message_id(),
            conversation_id=conversation_id,
            content=response_text,
            sender="assistant",
            timestamp=datetime.utcnow()
        )
        db.add(assistant_message)
        
        # Create new conversation if needed
        if not request.conversation_id:
            title = generate_conversation_title(request.message, model)
            conversation = Conversation(
                id=conversation_id,
                title=title,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            db.add(conversation)
        else:
            # Update conversation timestamp
            db.query(Conversation).filter(
                Conversation.id == conversation_id
            ).update({"updated_at": datetime.utcnow()})
        
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
    try:
        # Get conversations with their last message for preview
        conversations = db.query(Conversation).order_by(desc(Conversation.updated_at)).all()
        
        # Get last message for each conversation
        conversation_responses = []
        for conv in conversations:
            last_message = db.query(Message).filter(
                Message.conversation_id == conv.id
            ).order_by(desc(Message.timestamp)).first()
            
            preview = last_message.content[:50] + "..." if last_message and len(last_message.content) > 50 else (
                last_message.content if last_message else None
            )
            
            conversation_responses.append(ConversationResponse(
                id=conv.id,
                title=conv.title,
                created_at=conv.created_at,
                updated_at=conv.updated_at,
                preview=preview
            ))
        
        return conversation_responses
    except Exception as e:
        logger.error(f"Error fetching conversations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/conversations/{conversation_id}/messages", response_model=List[MessageHistoryResponse])
async def get_conversation_messages(
    conversation_id: str,
    db: Session = Depends(get_db)
):
    try:
        # Verify conversation exists
        conversation = db.query(Conversation).filter(
            Conversation.id == conversation_id
        ).first()
        
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        messages = db.query(Message).filter(
            Message.conversation_id == conversation_id
        ).order_by(Message.timestamp.asc()).all()
        
        return messages
    except Exception as e:
        logger.error(f"Error fetching messages: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    db: Session = Depends(get_db)
):
    try:
        # Delete messages first
        db.query(Message).filter(
            Message.conversation_id == conversation_id
        ).delete()
        
        # Then delete conversation
        db.query(Conversation).filter(
            Conversation.id == conversation_id
        ).delete()
        
        db.commit()
        return {"status": "success", "message": "Conversation deleted"}
    except Exception as e:
        db.rollback()
        logger.error(f"Error deleting conversation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
