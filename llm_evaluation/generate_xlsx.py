import pandas as pd

filename = input("Enter file name: ")
df = pd.read_json(filename)
df['cost'] = df['llm_cost'].apply(lambda x:x['upstream_inference_cost'])
df['total_tokens'] = df['llm_cost'].apply(lambda x:x['total_tokens'])
df = df.rename({
    "llm_references": "retrieved_contexts",
    "llm_response": "response"
}, axis=1)
df = df[["user_input","retrieved_contexts","reference_contexts","response","reference","cost","total_tokens"]]
df.to_excel(filename.replace(".json",'.xlsx'), index=False)
# print(df.head())