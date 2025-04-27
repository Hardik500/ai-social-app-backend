import re
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class WhatsAppParser:
    """
    Parser for WhatsApp exported data.
    Parses WhatsApp .txt export format.
    """
    
    @staticmethod
    def parse(data: Any) -> List[Dict[str, Any]]:
        """
        Parse WhatsApp data to extract messages.
        Args:
            data: WhatsApp export data (bytes or str)
        Returns:
            List of dictionaries with message data: {user_id, timestamp, text}
        """
        if isinstance(data, bytes):
            text = data.decode('utf-8')
        else:
            text = str(data)
        
        # WhatsApp message regex: [date, time] sender: message
        # Example: [26/4/25, 4:32:21	PM] Shambhavi Tewari: Yesss
        msg_pattern = re.compile(r"^\[(\d{1,2}/\d{1,2}/\d{2,4}), (\d{1,2}:\d{2}:\d{2}\s*[APMapm\.]*?)\] ([^:]+): (.*)$")
        
        messages = []
        current_msg = None
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            match = msg_pattern.match(line)
            if match:
                # Save previous message
                if current_msg:
                    messages.append(current_msg)
                date_str, time_str, sender, message = match.groups()
                # Parse timestamp
                try:
                    # WhatsApp date: d/m/yy or d/m/yyyy, time: h:mm:ss AM/PM
                    dt = WhatsAppParser._parse_datetime(date_str, time_str)
                    timestamp = dt.isoformat()
                except Exception:
                    timestamp = f"{date_str} {time_str}"
                current_msg = {
                    "user_id": sender.strip(),
                    "timestamp": timestamp,
                    "text": message.strip()
                }
            else:
                # Continuation of previous message
                if current_msg:
                    current_msg["text"] += "\n" + line
        # Add last message
        if current_msg:
            messages.append(current_msg)
        return messages

    @staticmethod
    def _parse_datetime(date_str: str, time_str: str) -> datetime:
        # Try parsing with and without AM/PM
        for fmt in ["%d/%m/%y, %I:%M:%S %p", "%d/%m/%Y, %I:%M:%S %p", "%d/%m/%y, %H:%M:%S", "%d/%m/%Y, %H:%M:%S"]:
            try:
                return datetime.strptime(f"{date_str}, {time_str}", fmt)
            except Exception:
                continue
        # Fallback: try without comma
        for fmt in ["%d/%m/%y %I:%M:%S %p", "%d/%m/%Y %I:%M:%S %p", "%d/%m/%y %H:%M:%S", "%d/%m/%Y %H:%M:%S"]:
            try:
                return datetime.strptime(f"{date_str} {time_str}", fmt)
            except Exception:
                continue
        raise ValueError(f"Unrecognized date/time format: {date_str} {time_str}")

    @staticmethod
    def _extract_message_data(message: Any) -> Optional[Dict[str, Any]]:
        # Not used in this implementation
        return None 