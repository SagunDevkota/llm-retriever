import json
from pathlib import Path


ROUTING_INPUT_PRICE_PER_M = 0.03
ROUTING_OUTPUT_PRICE_PER_M = 0.32

GENERATION_INPUT_PRICE_PER_M = 0.25
GENERATION_OUTPUT_PRICE_PER_M = 1.50


def average_tokens_and_cost(file_path: str) -> dict:
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    routing_input_tokens = []
    routing_output_tokens = []
    generation_input_tokens = []
    generation_output_tokens = []

    for item in data:
        llm_cost = item.get("llm_cost", {})

        # Routing
        if "routing" in llm_cost:
            routing = llm_cost["routing"]

            routing_input_tokens.append(
                routing.get("prompt_tokens") or 0
            )
            routing_output_tokens.append(
                routing.get("completion_tokens") or 0
            )
        else:
            routing_input_tokens.append(0)
            routing_output_tokens.append(0)

        # Generation
        if "generation" in llm_cost:
            generation = llm_cost["generation"]

            generation_input_tokens.append(
                generation.get("prompt_tokens") or 0
            )
            generation_output_tokens.append(
                generation.get("completion_tokens") or 0
            )
        else:
            # Flat/non-nested llm_cost:
            # prompt_tokens     -> generation input
            # completion_tokens -> generation output
            generation_input_tokens.append(
                llm_cost.get("prompt_tokens") or 0
            )
            generation_output_tokens.append(
                llm_cost.get("completion_tokens") or 0
            )

    count = len(data)

    if count == 0:
        raise ValueError(f"{file_path} contains no records.")

    avg_routing_input = sum(routing_input_tokens) / count
    avg_routing_output = sum(routing_output_tokens) / count
    avg_generation_input = sum(generation_input_tokens) / count
    avg_generation_output = sum(generation_output_tokens) / count

    # Cost per average request
    routing_input_cost = (
        avg_routing_input / 1_000_000
    ) * ROUTING_INPUT_PRICE_PER_M

    routing_output_cost = (
        avg_routing_output / 1_000_000
    ) * ROUTING_OUTPUT_PRICE_PER_M

    generation_input_cost = (
        avg_generation_input / 1_000_000
    ) * GENERATION_INPUT_PRICE_PER_M

    generation_output_cost = (
        avg_generation_output / 1_000_000
    ) * GENERATION_OUTPUT_PRICE_PER_M

    routing_cost = routing_input_cost + routing_output_cost
    generation_cost = generation_input_cost + generation_output_cost
    total_cost = routing_cost + generation_cost

    return {
        "records": count,

        "average_routing_input_tokens": avg_routing_input,
        "average_routing_output_tokens": avg_routing_output,
        "average_generation_input_tokens": avg_generation_input,
        "average_generation_output_tokens": avg_generation_output,

        "average_routing_input_cost": routing_input_cost,
        "average_routing_output_cost": routing_output_cost,
        "average_routing_cost": routing_cost,

        "average_generation_input_cost": generation_input_cost,
        "average_generation_output_cost": generation_output_cost,
        "average_generation_cost": generation_cost,

        "average_total_cost": total_cost,

        "average_input_tokens": avg_routing_input + avg_generation_input,
        "average_output_tokens": avg_routing_output + avg_generation_output,
        "average_total_tokens": (
            avg_routing_input + avg_routing_output
            + avg_generation_input + avg_generation_output
        ),
    }


def print_results(name: str, results: dict) -> None:
    print(f"\n{name}")
    print("=" * 60)

    print("\nTOKENS")
    print("-" * 60)
    print(
        f"Average routing input tokens:     "
        f"{results['average_routing_input_tokens']:,.2f}"
    )
    print(
        f"Average routing output tokens:    "
        f"{results['average_routing_output_tokens']:,.2f}"
    )
    print(
        f"Average generation input tokens:  "
        f"{results['average_generation_input_tokens']:,.2f}"
    )
    print(
        f"Average generation output tokens: "
        f"{results['average_generation_output_tokens']:,.2f}"
    )

    print("\nCOST PER REQUEST")
    print("-" * 60)
    print(
        f"Routing cost:      "
        f"${results['average_routing_cost']:.8f}"
    )
    print(
        f"Generation cost:   "
        f"${results['average_generation_cost']:.8f}"
    )
    print(
        f"Average total cost:"
        f" ${results['average_total_cost']:.8f}"
    )


files = {
    "llm_retriever": "questions_llm_retriever.json",
    "raptor": "questions_raptor.json",
    "naive": "questions_naive_rag.json",
    "hybrid": "questions_hybrid_rerank_playwright.json",
    "ketrag": "questions_ketrag_playwright.json",
    "rankgpt": "questions_rankgpt.json",
}


results = {}

for name, file_path in files.items():
    if not Path(file_path).exists():
        print(f"\nSkipping {name}: {file_path} not found")
        continue
    results[name] = average_tokens_and_cost(file_path)
    print_results(name, results[name])


# ---------------------------------------------------------
# Comparison:
#
# cheapest method = 1x, every other method = cost / cheapest
# ---------------------------------------------------------

costs = {
    name: result["average_total_cost"]
    for name, result in results.items()
}

cheapest = min(costs, key=costs.get)
cheapest_cost = costs[cheapest]

ranking = [
    {
        "method": name,
        "average_cost": cost,
        "cost_multiple": cost / cheapest_cost,
    }
    for name, cost in sorted(costs.items(), key=lambda kv: kv[1])
]

tokens = {
    name: result["average_total_tokens"]
    for name, result in results.items()
}

fewest = min(tokens, key=tokens.get)
fewest_tokens = tokens[fewest]

token_ranking = [
    {
        "method": name,
        "average_input_tokens": results[name]["average_input_tokens"],
        "average_output_tokens": results[name]["average_output_tokens"],
        "average_total_tokens": total,
        "token_multiple": total / fewest_tokens,
    }
    for name, total in sorted(tokens.items(), key=lambda kv: kv[1])
]

comparison = {
    "baseline": cheapest,
    "baseline_average_cost": cheapest_cost,
    "formula": "method_cost / min(method_costs)",
    "ranking": ranking,
    "token_baseline": fewest,
    "token_baseline_average_tokens": fewest_tokens,
    "token_formula": "method_total_tokens / min(method_total_tokens)",
    "token_ranking": token_ranking,
}

results["comparison"] = comparison


print("\nCOMPARISON (relative to cheapest)")
print("=" * 60)
for row in ranking:
    print(
        f"{row['method']:<16} ${row['average_cost']:.8f}"
        f"   {row['cost_multiple']:.4f}x"
    )


print("\nTOKEN COMPARISON (relative to fewest total tokens)")
print("=" * 60)
print(f"{'method':<16} {'input':>12} {'output':>10} {'total':>12}  multiple")
for row in token_ranking:
    print(
        f"{row['method']:<16}"
        f" {row['average_input_tokens']:>12,.2f}"
        f" {row['average_output_tokens']:>10,.2f}"
        f" {row['average_total_tokens']:>12,.2f}"
        f"  {row['token_multiple']:.4f}x"
    )


# ---------------------------------------------------------
# Write all results to file
# ---------------------------------------------------------

output_file = "token_cost_summary.json"

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print(f"\nResults written to: {output_file}")