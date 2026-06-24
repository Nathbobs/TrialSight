# SKILLS.md — TrialSight Capabilities Reference

## ClinicalTrials.gov API
- Base URL: https://clinicaltrials.gov/api/v2/studies
- Key params: query.cond (condition), pageSize (max 1000), format (json)
- Example: GET /studies?query.cond=cancer&pageSize=100&format=json
- Response: JSON with studies array, each study has protocolSection
- No API key required
- Rate limit: be respectful, add small sleep between paginated calls

## XML/JSON Parsing
- API returns JSON by default — use requests + json()
- For XML fallback: use lxml.etree or xml.etree.ElementTree
- Key nested path: study → protocolSection → identificationModule → nctId
- Always use .get() with defaults to handle missing fields safely

## Pandas ETL Operations
- Read/write Parquet: pd.read_parquet(), df.to_parquet()
- Null handling: df.dropna(subset=[critical_cols])
- Date normalization: pd.to_datetime(df[col], errors='coerce')
- Always log: print(f"[LAYER] Input: {len(df_in)} → Output: {len(df_out)}")

## Qdrant Operations
- Client: from qdrant_client import QdrantClient
- Init: QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
- Create collection: client.recreate_collection(name, vectors_config)
- Upsert: client.upsert(collection_name, points=[PointStruct(...)])
- Search: client.search(collection_name, query_vector, limit=5)
- Always batch upserts in groups of 100

## HuggingFace Embeddings
- Model: sentence-transformers/all-MiniLM-L6-v2
- Init: from sentence_transformers import SentenceTransformer
- Usage: model.encode(text) → returns numpy array of 384 dims
- Convert to list for Qdrant: embedding.tolist()
- Cache model in st.cache_resource to avoid reloading

## Groq LLM Calls
- Client: from groq import Groq
- Init: Groq(api_key=GROQ_API_KEY)
- Model: llama-3.3-70b-versatile
- Usage: client.chat.completions.create(model, messages)
- Always include system prompt defining the clinical assistant role
- Max tokens: 1024 for chat responses

## Streamlit Patterns
- Secrets: st.secrets["GROQ_API_KEY"]
- Cache data: @st.cache_data for pipeline results
- Cache resource: @st.cache_resource for model and DB clients
- Chat: st.chat_message("user") / st.chat_message("assistant")
- Chat input: st.chat_input("Ask about clinical trials...")
- Session state: st.session_state.messages for chat history
- Tabs: tab1, tab2 = st.tabs(["Dashboard", "RAG Chat"])
