import os
from pathlib import Path
from typing import Dict, Any, Union, List

class PromptManager:
    """
    Manages loading and formatting of prompt templates.
    
    This class provides functionality to load prompt templates from files and
    format them with dynamic values.
    """
    
    def __init__(self, prompts_dir: str = None):
        """
        Initialize the PromptManager with the directory containing prompt templates.
        
        Args:
            prompts_dir: Path to the directory containing prompt templates.
                         If None, defaults to app/core/prompts/ relative to the current file.
        """
        if prompts_dir is None:
            # Default to the prompts directory within the app
            self.prompts_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                'core', 'prompts'
            )
        else:
            self.prompts_dir = prompts_dir
        
        self.templates = {}
        self._load_templates()
    
    def _load_templates(self):
        """Load all template files from the prompts directory."""
        prompts_path = Path(self.prompts_dir)
        for template_file in prompts_path.glob('*.txt'):
            template_name = template_file.stem
            with open(template_file, 'r') as f:
                self.templates[template_name] = f.read()
    
    def get_template(self, template_name: str) -> str:
        """
        Get the raw template content by name.
        
        Args:
            template_name: Name of the template (without .txt extension)
            
        Returns:
            The template content as a string
            
        Raises:
            KeyError: If the template does not exist
        """
        if template_name not in self.templates:
            raise KeyError(f"Template '{template_name}' not found in {self.prompts_dir}")
        return self.templates[template_name]
    
    def format_template(self, template_name: str, **kwargs) -> str:
        """
        Format a template with the provided values.
        
        Args:
            template_name: Name of the template (without .txt extension)
            **kwargs: Key-value pairs to substitute in the template
            
        Returns:
            The formatted template as a string
            
        Raises:
            KeyError: If the template does not exist
        """
        # Get the raw template
        template = self.get_template(template_name)
        
        # Special handling for personality_simulation template
        if template_name == "personality_simulation":
            # For this template, we'll use a different approach
            # First, prepare the replacement values
            formatted_kwargs = {}
            for key, value in kwargs.items():
                if isinstance(value, dict) and value:
                    # Format dict as comma-separated key-value pairs
                    formatted_kwargs[key] = ', '.join([f'{k}: {v}' for k, v in value.items()])
                elif isinstance(value, list) and value:
                    # Format list as comma-separated values
                    formatted_kwargs[key] = ', '.join(value)
                else:
                    formatted_kwargs[key] = value
            
            # Replace all format specifiers one by one
            formatted_template = template
            for key, value in formatted_kwargs.items():
                formatted_template = formatted_template.replace('{' + key + '}', str(value))
            
            return formatted_template
        else:
            # Use standard Python formatting for other templates
            # Handle special case for dict and list values
            formatted_kwargs = {}
            for key, value in kwargs.items():
                if isinstance(value, dict) and value:
                    # Format dict as comma-separated key-value pairs
                    formatted_kwargs[key] = ', '.join([f'{k}: {v}' for k, v in value.items()])
                elif isinstance(value, list) and value:
                    # Format list as comma-separated values
                    formatted_kwargs[key] = ', '.join(value)
                else:
                    formatted_kwargs[key] = value
            
            return template.format(**formatted_kwargs)

# Create a singleton instance
prompt_manager = PromptManager() 