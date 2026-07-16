"""Render diagram JSON → Mermaid / Markdown tree."""

from __future__ import annotations


def render_mermaid(diagram: dict) -> str:
    """Convert diagram JSON {type, title, nodes} to Mermaid text."""
    if not diagram or not isinstance(diagram, dict):
        return ""

    diagram_type = (diagram.get("type") or "mindmap").strip().lower()
    title = (diagram.get("title") or "").strip()
    nodes = diagram.get("nodes") or []

    if not nodes:
        return ""

    if diagram_type == "flowchart":
        return _render_flowchart(title, nodes)
    elif diagram_type == "architecture":
        return _render_architecture(title, nodes)
    else:  # mindmap (default)
        return _render_mindmap(title, nodes)


def render_markdown_tree(diagram: dict) -> str:
    """Fallback: convert diagram JSON to hierarchical Markdown."""
    if not diagram or not isinstance(diagram, dict):
        return ""
    title = (diagram.get("title") or "会议结构图").strip()
    nodes = diagram.get("nodes") or []
    if not nodes:
        return f"# {title}\n\n(无节点)"

    # Build parent→children map
    children: dict[str, list[dict]] = {}
    roots = []
    for n in nodes:
        pid = n.get("parent") or ""
        if not pid:
            roots.append(n)
        else:
            children.setdefault(pid, []).append(n)

    lines = [f"# {title}", ""]
    for root in roots:
        _render_node_md(root, children, lines, 0)
    return "\n".join(lines)


def _render_node_md(node: dict, children: dict, lines: list, depth: int):
    indent = "  " * depth
    label = node.get("label", node.get("id", "?"))
    lines.append(f"{indent}- {label}")
    for child in children.get(node.get("id", ""), []):
        _render_node_md(child, children, lines, depth + 1)


def _escape_mermaid(text: str) -> str:
    """Escape special chars for Mermaid node labels."""
    return str(text).replace('"', "'").replace("\n", " ").replace("(", "[").replace(")", "]")


def _render_mindmap(title: str, nodes: list[dict]) -> str:
    lines = ["mindmap"]
    if title:
        lines.append(f"  root(({_escape_mermaid(title)}))")
    else:
        lines.append("  root((会议))")

    # Build parent→children and find roots
    children: dict[str, list[dict]] = {}
    roots = []
    for n in nodes:
        pid = n.get("parent") or ""
        if not pid:
            roots.append(n)
        else:
            children.setdefault(pid, []).append(n)

    for root in roots:
        _mermaid_mindmap_nodes(root, children, lines, 2)

    return "\n".join(lines)


def _mermaid_mindmap_nodes(node: dict, children: dict, lines: list, indent: int):
    prefix = "  " * indent
    label = _escape_mermaid(node.get("label", node.get("id", "?")))
    lines.append(f"{prefix}{label}")
    for child in children.get(node.get("id", ""), []):
        _mermaid_mindmap_nodes(child, children, lines, indent + 2)


def _render_flowchart(title: str, nodes: list[dict]) -> str:
    lines = ["flowchart TD"]
    if title:
        lines.append(f"  title[{_escape_mermaid(title)}]")

    edges = []
    for n in nodes:
        nid = n.get("id", "?")
        label = _escape_mermaid(n.get("label", nid))
        lines.append(f"  {nid}[{label}]")
        pid = n.get("parent") or ""
        if pid:
            edges.append(f"  {pid} --> {nid}")
        for child_id in n.get("children", []):
            edges.append(f"  {nid} --> {child_id}")

    if edges:
        lines.append("")
        lines.extend(edges)

    return "\n".join(lines)


def _render_architecture(title: str, nodes: list[dict]) -> str:
    lines = ["flowchart LR"]
    if title:
        lines.append(f"  title[{_escape_mermaid(title)}]")

    edges = []
    for n in nodes:
        nid = n.get("id", "?")
        label = _escape_mermaid(n.get("label", nid))
        lines.append(f"  {nid}[{label}]")
        pid = n.get("parent") or ""
        if pid:
            edges.append(f"  {pid} --> {nid}")

    if edges:
        lines.append("")
        lines.extend(edges)

    return "\n".join(lines)
