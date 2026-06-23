# AGENTS.md — TrialSight RAG Pipeline

## Overview
The RAG pipeline operates as a sequential agent workflow.
Each step has a single responsibility and passes its output
to the next step. No step should be skipped.

## Pipeline Steps

### Step 1 — Query Intake
- Receive raw user question from Streamlit chat input
- Validate: reject empty strings
- Log: print(f"[RAG] Query received: {query[:50]}...")

### Step 2 — Query Embedding
- Tool: HuggingFace all-MiniLM-L6-v2
- Input: raw query string
- Output: 384-dimensional vector
- Cache: use st.cache_resource for the embedding model
- Never re-instantiate the model per query

### Step 3 — Qdrant Retrieval
- Tool: Qdrant Cloud similarity search
- Input: query vector
- Parameters: limit=5, score_threshold=0.45
- Output: list of scored results with payload
- If 0 results returned: trigger fallback (see TRUSTWORTHINESS.md)
- Log: print(f"[RAG] Retrieved {len(results)} chunks, top score: {results[0].score:.3f}")

### Step 4 — Context Assembly
- Take top 5 retrieved chunks
- Format each as:
  "Trial {nct_id}: {brief_title}\nSummary: {brief_summary}"
- Join with double newline separator
- Prepend with: "Relevant clinical trials from the database:"
- Total context must not exceed 3000 tokens

### Step 5 — Groq LLM Generation
- Model: llama-3.3-70b-versatile
- System prompt:
  "You are a clinical trial intelligence assistant.
   Answer questions using only the provided clinical trial context.
   Always cite the NCT ID of trials you reference.
   If the context does not contain enough information,
   say so honestly rather than speculating."
- Messages: [system, {context as user message}, {query}]
- Max tokens: 1024
- Temperature: 0.3 (low for factual clinical answers)

### Step 6 — Response + Citations
- Display assistant response in st.chat_message
- Below response, show expander: "Source Trials"
- List each retrieved trial: NCT ID, title, relevance score
- Append to st.session_state.messages for chat history

## Fallback Behavior
- If retrieval returns 0 results or all scores below threshold:
  return "I couldn't find relevant clinical trials for that query.
  Try rephrasing or asking about a specific condition or drug."
- Never pass empty context to Groq
- Never fabricate trial information
