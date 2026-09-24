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
                routing.get("prompt_tokens", 0)
            )
            routing_output_tokens.append(
                routing.get("completion_tokens", 0)
            )
        else:
            routing_input_tokens.append(0)
            routing_output_tokens.append(0)

        # Generation
        if "generation" in llm_cost:
            generation = llm_cost["generation"]

            generation_input_tokens.append(
                generation.get("prompt_tokens", 0)
            )
            generation_output_tokens.append(
                generation.get("completion_tokens", 0)
            )
        else:
            # Flat/non-nested llm_cost:
            # prompt_tokens     -> generation input
            # completion_tokens -> generation output
            generation_input_tokens.append(
                llm_cost.get("prompt_tokens", 0)
            )
            generation_output_tokens.append(
                llm_cost.get("completion_tokens", 0)
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
}


results = {}

for name, file_path in files.items():
    results[name] = average_tokens_and_cost(file_path)
    print_results(name, results[name])


# ---------------------------------------------------------
# Comparison:
#
# llm_retriever / ((raptor + naive) / 2)
# ---------------------------------------------------------

llm_retriever_cost = results["llm_retriever"]["average_total_cost"]
raptor_cost = results["raptor"]["average_total_cost"]
naive_cost = results["naive"]["average_total_cost"]

baseline_average_cost = (raptor_cost + naive_cost) / 2

cost_ratio = llm_retriever_cost / baseline_average_cost


comparison = {
    "llm_retriever_average_cost": llm_retriever_cost,
    "raptor_average_cost": raptor_cost,
    "naive_average_cost": naive_cost,
    "raptor_naive_average_cost": baseline_average_cost,
    "formula": "llm_retriever / ((raptor + naive) / 2)",
    "cost_ratio": cost_ratio,
}

results["comparison"] = comparison


print("\nCOMPARISON")
print("=" * 60)
print(f"LLM Retriever:        ${llm_retriever_cost:.8f}")
print(f"RAPTOR:               ${raptor_cost:.8f}")
print(f"Naive RAG:            ${naive_cost:.8f}")
print(f"RAPTOR + Naive avg:   ${baseline_average_cost:.8f}")
print("-" * 60)
print(f"LLM Retriever ratio:  {cost_ratio:.4f}x")


# ---------------------------------------------------------
# Write all results to file
# ---------------------------------------------------------

output_file = "token_cost_summary.json"

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print(f"\nResults written to: {output_file}")