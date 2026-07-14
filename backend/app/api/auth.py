from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from jose import jwt
from datetime import datetime, timedelta

router = APIRouter()

SECRET_KEY = "mysecretkey"
ALGORITHM = "HS256"

class LoginData(BaseModel):
    username: str
    password: str

fake_user = {
    "username": "admin",
    "password": "admin123"
}

@router.post("/login")
def login(data: LoginData):

    if (
        data.username != fake_user["username"]
        or data.password != fake_user["password"]
    ):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token_data = {
        "sub": data.username,
        "exp": datetime.utcnow() + timedelta(hours=5)
    }

    token = jwt.encode(token_data, SECRET_KEY, algorithm=ALGORITHM)

    return {
        "access_token": token,
        "token_type": "bearer"
    }
