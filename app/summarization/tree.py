"""Tree traversal / visualisation over a ``nodes_by_id`` dict."""


def post_order_traversal(nodes_dict):
    """
    Return nodes in post-order (children before parents) so that a parent is
    summarized only after all of its children have been.

    nodes_dict = {node_id: node_data}, where node_data has "parent_id" and
    "children" (a list of child ids).
    """
    result = []

    # Find root nodes
    root_nodes = [
        node_id
        for node_id, node in nodes_dict.items()
        if node["parent_id"] is None
    ]

    def dfs(node_id):
        node = nodes_dict[node_id]

        # Traverse children first
        for child_id in node["children"]:
            dfs(child_id)

        # Then visit current node
        result.append(node)

    for root_id in root_nodes:
        dfs(root_id)

    return result


def print_tree(nodes_by_id):
    # Find roots
    roots = [
        node
        for node in nodes_by_id.values()
        if node["parent_id"] is None
    ]

    def recurse(node, prefix="", is_last=True):
        connector = "└── " if is_last else "├── "

        print(f"{prefix}{connector}{node['title']}")

        children = node["children"]

        for i, child_id in enumerate(children):
            child = nodes_by_id[child_id]

            next_prefix = prefix + ("    " if is_last else "│   ")

            recurse(
                child,
                next_prefix,
                i == len(children) - 1,
            )

    for i, root in enumerate(roots):
        recurse(root, "", i == len(roots) - 1)
