import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

from sqlalchemy.orm import Session
from app.models.user import User
from app.models.conversation import Conversation, Message
from app.services.parsers.slack_parser import SlackParser
from app.services.parsers.whatsapp_parser import WhatsAppParser

logger = logging.getLogger(__name__)

class IngestionService:
    """
    Service to handle ingestion of data from various sources.
    Each source has its own parser implementation.
    """
    
    def __init__(self, db: Session):
        self.db = db
    
    def ingest_data(self, 
                   source_type: str, 
                   source_data: Dict[str, Any], 
                   primary_user_info: Dict[str, str],
                   additional_users: Optional[List[Dict[str, str]]] = None,
                   user_mapping: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Main entry point for data ingestion.
        
        Args:
            source_type: Type of the source (e.g., 'slack_har', 'whatsapp')
            source_data: Source data to be ingested
            primary_user_info: Information about the primary user
            additional_users: Information about additional users
            user_mapping: Optional mapping from source user IDs to usernames
            
        Returns:
            Dictionary with ingestion results
        """
        # Ensure users exist in the database
        primary_user = self._ensure_user_exists(primary_user_info)
        
        additional_users_db = []
        if additional_users:
            for user_info in additional_users:
                user = self._ensure_user_exists(user_info)
                additional_users_db.append(user)
        
        # Select parser based on source type
        if source_type.lower() == 'slack_har':
            return self._process_slack_data(source_data, primary_user, additional_users_db, user_mapping)
        elif source_type.lower() == 'whatsapp':
            return self._process_whatsapp_data(source_data, primary_user, additional_users_db)
        else:
            return {"status": "error", "message": f"Unknown source type: {source_type}"}
    
    def _ensure_user_exists(self, user_info: Dict[str, str]) -> User:
        """
        Ensure a user exists in the database, create if not.
        
        Args:
            user_info: Dictionary with user information
            
        Returns:
            User object
        """
        username = user_info.get('username')
        email = user_info.get('email')
        phone = user_info.get('phone')
        description = user_info.get('description')
        
        # Try to find existing user
        user = None
        if username:
            user = self.db.query(User).filter(User.username == username).first()
        
        if not user and email:
            user = self.db.query(User).filter(User.email == email).first()
            
        if not user and phone:
            user = self.db.query(User).filter(User.phone == phone).first()
        
        # Create new user if not found
        if not user:
            user = User(
                username=username,
                email=email,
                phone=phone,
                description=description
            )
            self.db.add(user)
            self.db.commit()
            self.db.refresh(user)
        
        return user
    
    def _process_slack_data(self, 
                          har_data: Dict[str, Any], 
                          primary_user: User,
                          additional_users: List[User] = None,
                          user_mapping: Dict[str, str] = None) -> Dict[str, Any]:
        """
        Process Slack HAR data using the dedicated parser.
        
        Args:
            har_data: Slack HAR data
            primary_user: Primary user object
            additional_users: Additional user objects
            user_mapping: Mapping from Slack user IDs to usernames
            
        Returns:
            Dictionary with processing results
        """
        try:
            # Create a new conversation
            conversation = Conversation(
                source="slack",
                user_id=primary_user.id
            )
            self.db.add(conversation)
            self.db.commit()
            self.db.refresh(conversation)
            
            # Initialize user mapping if none provided
            if user_mapping is None:
                user_mapping = {}
            
            # Use the SlackParser to extract messages
            parsed_messages = SlackParser.parse(har_data)
            
            # Map of user IDs/usernames to database user IDs
            username_to_db_id = {primary_user.username: primary_user.id}
            if additional_users:
                for user in additional_users:
                    username_to_db_id[user.username] = user.id
            
            # Store parsed messages in the database
            messages_imported = 0
            unknown_users = set()
            
            for parsed_msg in parsed_messages:
                slack_user_id = parsed_msg.get('user_id')
                
                # Map the Slack user ID to a username if mapping exists
                username = user_mapping.get(slack_user_id)
                
                # If no mapping exists, use the ID as username
                if username is None:
                    unknown_users.add(slack_user_id)
                    username = f"unknown_{slack_user_id}"
                
                # Get or create user
                if username in username_to_db_id:
                    db_user_id = username_to_db_id[username]
                else:
                    # Create user if not found
                    user = self._get_user_by_username_or_create(username)
                    db_user_id = user.id
                    username_to_db_id[username] = db_user_id
                
                # Create message
                message = Message(
                    conversation_id=conversation.id,
                    user_id=db_user_id,
                    content=parsed_msg.get('text', ''),
                    timestamp=parsed_msg.get('timestamp', '')
                )
                self.db.add(message)
                messages_imported += 1
            
            self.db.commit()
            
            result = {
                "status": "success",
                "conversation_id": conversation.id,
                "messages_imported": messages_imported
            }
            
            # Add warning if unknown users found
            if unknown_users:
                result["warnings"] = {
                    "unknown_users": list(unknown_users),
                    "message": "Some user IDs could not be mapped to usernames. Consider providing a user_mapping."
                }
            
            return result
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error processing Slack data: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def _process_whatsapp_data(self, 
                             data: Any, 
                             primary_user: User,
                             additional_users: List[User] = None) -> Dict[str, Any]:
        """
        Process WhatsApp data using the dedicated parser.
        
        Args:
            data: WhatsApp data
            primary_user: Primary user object
            additional_users: Additional user objects
            
        Returns:
            Dictionary with processing results
        """
        try:
            # Create a new conversation
            conversation = Conversation(
                source="whatsapp",
                user_id=primary_user.id
            )
            self.db.add(conversation)
            self.db.commit()
            self.db.refresh(conversation)
            
            # Use the WhatsAppParser to extract messages
            parsed_messages = WhatsAppParser.parse(data)
            
            # Store parsed messages in the database
            messages_imported = 0
            for parsed_msg in parsed_messages:
                # Implementation will be completed when WhatsApp format is known
                pass
                
            self.db.commit()
            
            return {
                "status": "success",
                "conversation_id": conversation.id,
                "messages_imported": messages_imported
            }
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error processing WhatsApp data: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def _get_user_by_username_or_create(self, username: str) -> User:
        """
        Find a user by username or create a new one.
        
        Args:
            username: Username to search for
            
        Returns:
            User object
        """
        user = self.db.query(User).filter(User.username == username).first()
        
        if not user:
            user = User(
                username=username,
                email=f"{username}@example.com",  # Placeholder
                description="Imported from conversation data"
            )
            self.db.add(user)
            self.db.commit()
            self.db.refresh(user)
        
        return user 