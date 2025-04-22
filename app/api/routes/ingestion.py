from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
import json
from typing import Dict, List, Optional, Any

from app.db.database import get_db
from app.services.ingestion_service import IngestionService
from app.models.schemas import UserInfoSchema, AdditionalUserSchema

router = APIRouter(
    prefix="/ingestion",
    tags=["ingestion"],
    responses={404: {"description": "Not found"}},
)

@router.post("/")
async def ingest_data(
    source_type: str = Form(...),
    source_file: UploadFile = File(...),
    primary_user_info: str = Form(...),
    additional_users: Optional[str] = Form(None),
    user_mapping: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    """
    Ingest data from various sources.
    
    Args:
        source_type: Type of source (slack_har, whatsapp)
        source_file: Source file to ingest
        primary_user_info: JSON string with primary user information
        additional_users: Optional JSON string with additional users information
        user_mapping: Optional JSON string with mapping from source user IDs to usernames
    
    Returns:
        Ingestion results
    """
    
    try:
        # Parse JSON strings
        primary_user_data = json.loads(primary_user_info)
        additional_users_data = json.loads(additional_users) if additional_users else []
        user_mapping_data = json.loads(user_mapping) if user_mapping else {}
        
        # Read file content
        file_content = await source_file.read()
        
        # Parse as JSON if source_type is slack_har
        if source_type.lower() == 'slack_har':
            try:
                source_data = json.loads(file_content.decode('utf-8'))
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid JSON in source file")
        else:
            # For other source types, keep as bytes
            source_data = file_content
        
        # Create ingestion service and process data
        ingestion_service = IngestionService(db)
        result = ingestion_service.ingest_data(
            source_type=source_type,
            source_data=source_data,
            primary_user_info=primary_user_data,
            additional_users=additional_users_data,
            user_mapping=user_mapping_data
        )
        
        if result.get("status") == "error":
            raise HTTPException(status_code=400, detail=result.get("message", "Ingestion failed"))
        
        return result
        
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON format in user information")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/test")
async def test_ingestion(
    source_type: str,
    source_data: Dict[str, Any],
    primary_user_info: UserInfoSchema,
    additional_users: Optional[List[AdditionalUserSchema]] = None,
    user_mapping: Optional[Dict[str, str]] = None,
    db: Session = Depends(get_db)
):
    """
    Test endpoint for data ingestion (JSON data instead of file upload).
    
    Args:
        source_type: Type of source (slack_har, whatsapp)
        source_data: Source data to ingest
        primary_user_info: Primary user information
        additional_users: Additional users information
        user_mapping: Mapping from source user IDs to usernames
    
    Returns:
        Ingestion results
    """
    try:
        # Create ingestion service and process data
        ingestion_service = IngestionService(db)
        result = ingestion_service.ingest_data(
            source_type=source_type,
            source_data=source_data,
            primary_user_info=primary_user_info.dict(),
            additional_users=[user.dict() for user in additional_users] if additional_users else None,
            user_mapping=user_mapping
        )
        
        if result.get("status") == "error":
            raise HTTPException(status_code=400, detail=result.get("message", "Ingestion failed"))
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 