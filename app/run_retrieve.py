"""Phase 5 — LLM-guided retrieval (RAG).

Walks the node tree in Postgres level by level, letting the LLM pick the best
branches to explore, then generates and prints the final answer. Needs nodes
already stored in the `nodes` table, plus HETZNER_API_KEY (ROUTING model) and
OPENROUTER_API_KEY (ANSWER model) in the env.
    python run_retrieve.py --db <database_name>
"""

import argparse

from core.config import POSTGRES_DSN, dsn_with_db
from db.db2 import PostgresStore
from retrieval.retriever import LLMRetriever

import json
import tqdm
import pandas as pd

ap = argparse.ArgumentParser(description="Phase 5: LLM-guided retrieval (RAG).")
ap.add_argument(
    "--db", required=True,
    help="Database to query (required). Host/user/port come from POSTGRES_DSN.",
)
args = ap.parse_args()

# The database name is required and swapped into POSTGRES_DSN (host/user/port).
dsn = dsn_with_db(POSTGRES_DSN, args.db)

with open("./questions.json") as f:
    datasets = json.load(f)

store = PostgresStore(dsn)
# ROUTING model (Hetzner) for the tree walk, ANSWER model (OpenRouter) for the
# final generation; both ids and keys come from env / .env.
retriever = LLMRetriever()

# query = input("Enter your question: ")
# answer, selected_node_content = retriever.answer(query, store)
# print(answer,'\n\n\n', selected_node_content)

for dataset in tqdm.tqdm(datasets):
    if dataset.get('llm_response'):
        print(f"Skipping: {dataset['question_id']}")
        continue
    retrieved = retriever.answer(dataset['user_input'], store)
    answer, selected_node_content = retrieved
    if selected_node_content:
        dataset['llm_response'] = answer
        dataset['llm_references'] = selected_node_content
    else:
        dataset['llm_response'] = ""
        dataset['llm_references'] = []
    dataset['llm_cost'] = retriever.last_usage

    with open("./questions.json","w") as f:
        json.dump(datasets, f, indent=4)

df = pd.DataFrame(datasets)
# Rows answered before cost tracking existed have no llm_cost; default to 0.
df['cost'] = df.get('llm_cost', pd.Series(dtype=object)).apply(
    lambda x: (x or {}).get('upstream_inference_cost', 0) if isinstance(x, dict) else 0
)
df['total_tokens'] = df.get('llm_cost', pd.Series(dtype=object)).apply(
    lambda x: (x or {}).get('total_tokens', 0) if isinstance(x, dict) else 0
)
df = df.rename({
    "llm_references": "retrieved_contexts",
    "llm_response": "response"
}, axis=1)
df = df[["user_input","retrieved_contexts","reference_contexts","response","reference","cost","total_tokens"]]
df.to_excel("ragas_result.xlsx", index=False)
store.close()
