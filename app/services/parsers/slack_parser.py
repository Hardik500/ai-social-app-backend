import json
import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class SlackParser:
    """
    Parser for Slack HAR files.
    Extracts messages, users, and timestamps from Slack HAR file exports.
    """
    
    @staticmethod
    def parse(har_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Parse Slack HAR data to extract messages.
        
        Args:
            har_data: HAR file data
            
        Returns:
            List of dictionaries with message data
        """
        parsed_messages = []
        
        # Ensure we have valid HAR data
        if isinstance(har_data, str):
            try:
                har_data = json.loads(har_data)
            except json.JSONDecodeError:
                logger.error("Failed to parse HAR data as JSON")
                return []
        
        # Process entries in the HAR file
        entries = har_data.get('log', {}).get('entries', [])
        
        for entry in entries:
            # Look for response content that contains Slack messages
            response = entry.get('response', {})
            content = response.get('content', {})
            
            if not content:
                continue
                
            content_text = content.get('text', '')
            if not content_text:
                continue
            
            # Parse content text if it's a string
            if isinstance(content_text, str):
                try:
                    content_json = json.loads(content_text)
                except json.JSONDecodeError:
                    continue
            else:
                content_json = content_text
            
            # Extract messages if available
            messages = content_json.get('messages', [])
            if isinstance(messages, list):
                for message in messages:
                    parsed_message = SlackParser._extract_message_data(message)
                    if parsed_message:
                        parsed_messages.append(parsed_message)
            
            # Extract messages from messages_data if available
            messages_data = content_json.get('messages_data', {})
            if isinstance(messages_data, dict):
                for channel_data in messages_data.values():
                    channel_messages = channel_data.get('messages', [])
                    if isinstance(channel_messages, list):
                        for message in channel_messages:
                            parsed_message = SlackParser._extract_message_data(message)
                            if parsed_message:
                                parsed_messages.append(parsed_message)
        
        return parsed_messages
    
    @staticmethod
    def _extract_message_data(message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Extract relevant data from a Slack message.
        
        Args:
            message: Slack message data
            
        Returns:
            Dictionary with extracted data or None if invalid
        """
        # Check if this is a valid message
        if not message or not isinstance(message, dict):
            return None
            
        # Get basic message data
        user_id = message.get('user')
        timestamp = message.get('ts')
        text = message.get('text')
        
        # Skip if missing essential fields
        if not (user_id and timestamp and text is not None):
            return None
            
        # Extract additional data if available
        message_type = message.get('type', 'message')
        team = message.get('team')
        
        # Get thread information if available
        thread_ts = message.get('thread_ts')
        is_thread = thread_ts is not None
        
        # Extract reactions if available
        reactions = []
        if 'reactions' in message and isinstance(message['reactions'], list):
            for reaction in message['reactions']:
                reaction_name = reaction.get('name')
                reaction_count = reaction.get('count', 0)
                reaction_users = reaction.get('users', [])
                
                if reaction_name:
                    reactions.append({
                        'name': reaction_name,
                        'count': reaction_count,
                        'users': reaction_users
                    })
        
        # Extract attachments if available
        attachments = []
        if 'attachments' in message and isinstance(message['attachments'], list):
            for attachment in message['attachments']:
                attachments.append(attachment)
        
        # Return parsed message data
        return {
            'user_id': user_id,
            'timestamp': timestamp,
            'text': text,
            'type': message_type,
            'team': team,
            'is_thread': is_thread,
            'thread_ts': thread_ts,
            'reactions': reactions,
            'attachments': attachments,
            'raw_data': message  # Include raw data for future reference
        } 