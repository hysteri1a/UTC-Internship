from typing import Optional
from pydantic import BaseModel

# Pydantic models
# Pydantic models
class CreateSessionRequest(BaseModel):
    pass

class SessionResponse(BaseModel):
    session_id: str

class FileUploadRequest(BaseModel):
    session_id: str
    file_type: str

class FileUploadResponse(BaseModel):
    file_type: str
    status: str
    name: str
    size: int

class PrinciplesRequest(BaseModel):
    session_id: str
    score_type: str

class SessionListRequest(BaseModel):
    pass

class SessionListItem(BaseModel):
    session_id: str
    created_at: str
    file_count: int

class SessionDetailRequest(BaseModel):
    session_id: str
    field: Optional[str] = None

class DeleteFileRequest(BaseModel):
    session_id: str
    file_type: str

class DeleteSessionRequest(BaseModel):
    session_id: str

class DeleteResponse(BaseModel):
    status: str
    session_id: Optional[str] = None
    file_type: Optional[str] = None

class ScoreRequest(BaseModel):
    session_id: str
    score_type: str  # 新增参数
    enable_image_recognition: Optional[bool] = False  # 默认为 True，保持向后兼容性

class ScoringRequest(BaseModel):
    session_id: str