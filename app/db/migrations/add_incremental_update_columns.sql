-- Add incremental update columns to personality_profiles table
-- This migration adds support for incremental personality updates

-- Add last_message_id column
ALTER TABLE personality_profiles 
ADD COLUMN IF NOT EXISTS last_message_id INTEGER;

-- Add change_log column for tracking personality changes over time
ALTER TABLE personality_profiles 
ADD COLUMN IF NOT EXISTS change_log JSONB;

-- Add delta_embeddings column for storing incremental change embeddings
ALTER TABLE personality_profiles 
ADD COLUMN IF NOT EXISTS delta_embeddings JSONB;

-- Create index on last_message_id for faster lookups
CREATE INDEX IF NOT EXISTS idx_personality_profiles_last_message_id 
ON personality_profiles(last_message_id);

-- Add comment explaining the purpose of these columns
COMMENT ON COLUMN personality_profiles.last_message_id IS 'ID of the last message processed for incremental updates';
COMMENT ON COLUMN personality_profiles.change_log IS 'JSON log of personality changes over time';
COMMENT ON COLUMN personality_profiles.delta_embeddings IS 'JSON array of embeddings for personality changes'; 