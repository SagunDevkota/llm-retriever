"""Phase 5 — LLM-guided retrieval (RAG).

Walks the node tree in Postgres level by level, letting the LLM pick the best
branches to explore, then generates and prints the final answer. Needs nodes
already stored in the `nodes` table and OPENROUTER_API_KEY in the env.
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
retriever = LLMRetriever()  # reads OPENROUTER_API_KEY from env / .env

query = input("Enter your question: ")
answer, selected_node_content = retriever.answer(query, store)
print(answer,'\n\n\n', selected_node_content)

# for dataset in tqdm.tqdm(datasets):
#     if dataset.get('llm_response'):
#         print(f"Skipping: {dataset['question_id']}")
#         continue
#     retrieved = retriever.answer(dataset['user_input'], store)
#     answer, selected_node_content = retrieved
#     if selected_node_content:
#         dataset['llm_response'] = answer
#         dataset['llm_references'] = selected_node_content
#     else:
#         dataset['llm_response'] = ""
#         dataset['llm_references'] = []

#     with open("./questions.json","w") as f:
#         json.dump(datasets, f, indent=4)

# df = pd.DataFrame(datasets)
# df = df.rename({
#     "llm_references": "retrieved_contexts",
#     "llm_response": "response"
# }, axis=1)
# df = df[["user_input","retrieved_contexts","reference_contexts","response","reference"]]
# df.to_excel("ragas_result.xlsx", index=False)
# store.close()
