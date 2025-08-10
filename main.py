import os
from fastapi import FastAPI, HTTPException, Depends, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from typing import List, Optional
import google.generativeai as genai
from datetime import datetime, timedelta
from sqlalchemy import create_engine, Column, String, Integer, DateTime, Text, desc, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
import logging
import uuid
from jose import JWTError, jwt
from passlib.context import CryptContext
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Gemini Chatbot API",
    description="API for the Gemini-powered chatbot with conversation history and authentication",
    version="0.3.0",
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

# Security configuration
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")

# Database models
class User(Base):
    __tablename__ = "users"
    
    id = Column(String, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    name = Column(String)
    hashed_password = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    conversations = relationship("Conversation", back_populates="user")

class Conversation(Base):
    __tablename__ = "conversations"
    
    id = Column(String, primary_key=True, index=True)
    title = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    user_id = Column(String, ForeignKey("users.id"))
    
    user = relationship("User", back_populates="conversations")
    messages = relationship("Message", back_populates="conversation")

class Message(Base):
    __tablename__ = "messages"
    
    id = Column(String, primary_key=True, index=True)
    conversation_id = Column(String, ForeignKey("conversations.id"))
    content = Column(Text)
    sender = Column(String)  # 'user' or 'assistant'
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    conversation = relationship("Conversation", back_populates="messages")

# Create tables
Base.metadata.create_all(bind=engine)

# Pydantic models
class UserBase(BaseModel):
    email: str
    name: Optional[str] = None

class UserCreate(UserBase):
    password: str

class UserInDB(UserBase):
    id: str
    created_at: datetime
    
    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    email: Optional[str] = None

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
    preview: Optional[str] = None

class MessageHistoryResponse(BaseModel):
    id: str
    content: str
    sender: str
    timestamp: datetime

class GoogleAuthRequest(BaseModel):
    token: str
    email: str
    name: str

# Helper functions
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
        token_data = TokenData(email=email)
    except JWTError:
        raise credentials_exception
    
    user = db.query(User).filter(User.email == token_data.email).first()
    if user is None:
        raise credentials_exception
    return user

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

# Gemini configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyBIZbO5KawONRW9-JCBoIQ7vX5EhSKFhNM")
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-2.5-flash')

# Authentication endpoints
@app.post("/auth/signup", response_model=Token)
async def signup(user_data: UserCreate, db: Session = Depends(get_db)):
    # Check if user already exists
    existing_user = db.query(User).filter(User.email == user_data.email).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Create new user
    hashed_password = get_password_hash(user_data.password)
    user_id = f"user-{uuid.uuid4()}"
    new_user = User(
        id=user_id,
        email=user_data.email,
        name=user_data.name,
        hashed_password=hashed_password,
        created_at=datetime.utcnow()
    )
    
    db.add(new_user)
    db.commit()
    
    # Generate token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user_data.email}, expires_delta=access_token_expires
    )
    
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/auth/login", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )
    
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/auth/google", response_model=Token)
async def google_auth(google_data: GoogleAuthRequest, db: Session = Depends(get_db)):
    try:
        # Verify Google token
        google_response = requests.get(
            f"https://www.googleapis.com/oauth2/v3/tokeninfo?access_token={google_data.token}"
        )
        google_response.raise_for_status()
        token_info = google_response.json()
        
        if token_info.get("email") != google_data.email:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid Google token"
            )
        
        # Check if user exists
        user = db.query(User).filter(User.email == google_data.email).first()
        
        if not user:
            # Create new user
            user_id = f"user-{uuid.uuid4()}"
            new_user = User(
                id=user_id,
                email=google_data.email,
                name=google_data.name,
                hashed_password=get_password_hash(str(uuid.uuid4())),  # Random password for Google users
                created_at=datetime.utcnow()
            )
            db.add(new_user)
            db.commit()
            user = new_user
        
        # Generate token
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user.email}, expires_delta=access_token_expires
        )
        
        return {"access_token": access_token, "token_type": "bearer"}
    
    except requests.RequestException as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Google token"
        )

# API endpoints
@app.post("/chat", response_model=MessageResponse)
async def chat_with_gemini(
    request: MessageRequest,
    current_user: User = Depends(get_current_user),
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
                updated_at=datetime.utcnow(),
                user_id=current_user.id
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
async def get_conversations(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        # Get conversations with their last message for preview
        conversations = db.query(Conversation).filter(
            Conversation.user_id == current_user.id
        ).order_by(desc(Conversation.updated_at)).all()
        
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

@app.get("/{conversation_id}/messages", response_model=List[MessageHistoryResponse])
async def get_conversation_messages(
    conversation_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        # Verify conversation exists and belongs to user
        conversation = db.query(Conversation).filter(
            Conversation.id == conversation_id,
            Conversation.user_id == current_user.id
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

@app.delete("/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        # Verify conversation belongs to user
        conversation = db.query(Conversation).filter(
            Conversation.id == conversation_id,
            Conversation.user_id == current_user.id
        ).first()
        
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
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
