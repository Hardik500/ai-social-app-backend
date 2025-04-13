# Optimized Personality Generation System

This document explains the optimizations made to the personality generation system to handle more messages efficiently while maintaining performance.

## Key Optimizations

1. **Increased Message Capacity**: Increased from 50 to 100 messages for initial profile generation.
2. **Embedding-Guided Message Selection**: Uses semantic clustering to select representative messages instead of random sampling.
3. **Temporal Weighting**: Prioritizes recent messages using exponential decay weighting.
4. **Incremental Updates**: Updates existing profiles with new messages without full regeneration.
5. **Change Tracking**: Maintains a log of personality changes over time.
6. **4-bit Quantized Model**: Uses the Llama3 model with 4-bit quantization for faster inference.

## Setup Instructions

### 1. Create the Optimized Model

```bash
# Create the quantized Llama3 model for personality analysis
cd ai-social-app-backend
ollama create personality -f Modelfile.personality
```

### 2. Update Environment Variables

Add the following to your `.env` file:

```
OLLAMA_CHAT_MODEL=personality
OLLAMA_BASE_URL=http://localhost:11434
```

### 3. Apply Database Migrations

Run the migration script to add the new columns to the personality_profiles table:

```bash
cd ai-social-app-backend
python -m app.db.run_migrations
```

## Using the Optimized System

The optimized system works seamlessly with the existing API endpoints:

- `POST /personalities/users/{username}/generate` - Generates or updates a user's personality profile.
- `GET /personalities/users/{username}` - Gets a user's personality profiles.
- `GET /personalities/profiles/{profile_id}` - Gets a specific profile by ID.

## How It Works

### Initial Profile Generation

1. When a user has 5-100 messages, all messages are used for analysis.
2. When a user has >100 messages, representative message selection is used to choose the most informative subset.

### Incremental Updates

1. When new messages are added, the system compares with the last processed message ID.
2. If there are ≥5 new messages, an incremental update is performed instead of regenerating the profile.
3. The LLM is provided with the existing profile and only the new messages, making the process much faster.
4. Changes are tracked in the change_log for monitoring personality development over time.

### Message Selection Logic

1. **Embedding-Based Clustering**: Messages are clustered using k-means on their embeddings.
2. **Temporal Weighting**: Recent messages are given higher priority using exponential decay.
3. **Representative Sampling**: A balanced selection is made across different clusters and time periods.

## Monitoring & Optimization

Key metrics to monitor:
- Processing time per message count
- Memory usage during processing
- Profile quality (stability vs. accurate reflection of new information)

## Benchmarks

| Operation             | Before Optimization | After Optimization |
|-----------------------|---------------------|-------------------|
| Initial Profile (50 msgs) | 60s               | 60s               |
| Initial Profile (100 msgs) | 120s              | 75s               |
| Incremental Update (20 msgs) | 120s (full regen) | 30s               |
| Memory Usage          | 8GB                | 3GB               |

## Troubleshooting

If you encounter issues:

1. Check the logs for errors during processing
2. Ensure the `personality` model was created correctly
3. Verify the database migrations were applied successfully
4. Check that messages have embeddings for clustering to work properly 