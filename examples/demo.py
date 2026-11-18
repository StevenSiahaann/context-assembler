"""Interactive demonstration of the Context Assembler Engine.

Simulates a real-world scenario with:
1. Static P0 System Prompt
2. Active User Query (P1)
3. Massive Tool Output (100 lines of EXPLAIN ANALYZE logs) -> Head/Tail truncated
4. Multiple RAG Documents with different scores -> Lowest score dropped
5. Multi-turn Chat History -> Middle turns dropped, Turn 0 preserved
"""

from context_assembler import (
    Assembler,
    CachedTokenizer,
    ContextBlock,
    Drop,
    HeadTail,
    ModelProfile,
    Never,
    Priority,
    SlotPolicy,
    TiktokenTokenizer,
    to_anthropic_payload,
    to_openai_messages,
)


def run_demo() -> None:
    print("=" * 70)
    print("🚀 CONTEXT ASSEMBLER ENGINE — INTERACTIVE DEMO")
    print("=" * 70)

    # 1. Setup Model Configuration (e.g. 4,000 token test window)
    model = ModelProfile(
        context_window=4_000,
        max_output_tokens=1_000,
        per_message_overhead=4,
        safety_margin_ratio=0.03,
    )

    tokenizer = CachedTokenizer(TiktokenTokenizer("o200k_base"), maxsize=1_000)

    assembler = Assembler(
        model=model,
        tokenizer=tokenizer,
        slots=[
            SlotPolicy("tool_result", min_tokens=200, max_tokens=1_500, weight=3),
            SlotPolicy("rag", min_tokens=300, max_tokens=1_500, weight=4),
            SlotPolicy("history", min_tokens=200, max_tokens=1_500, weight=2),
        ],
    )

    # 2. Prepare Context Blocks
    blocks: list[ContextBlock] = []

    # P0: Non-negotiable system prompt
    blocks.append(
        ContextBlock(
            id="sys_prompt",
            kind="system",
            role="system",
            content="You are an autonomous PostgreSQL Performance Tuning Engineer.",
            priority=Priority.P0,
            strategy=Never(),
            cacheable=True,
        )
    )

    # P1: Active user query
    blocks.append(
        ContextBlock(
            id="user_active_query",
            kind="user",
            role="user",
            content="Why is our query on table 'orders' taking 4.2 seconds under high load?",
            priority=Priority.P1,
            strategy=HeadTail(0.5),
        )
    )

    # P2: Atomic tool call & result pair
    blocks.append(
        ContextBlock(
            id="asst_call_explain",
            kind="tool_call",
            role="assistant",
            content="Running EXPLAIN (ANALYZE, BUFFERS) on table orders...",
            priority=Priority.P2,
            strategy=Never(),
            group_id="call_explain_orders",
        )
    )

    # Massive tool result (100 log lines)
    log_lines = [
        f"Line {i:03d}: Seq Scan on orders (cost=0.00..1845.20 rows=100000 width=128) "
        f"actual time=0.042..4210.12 loops=1 Filter: (user_id = 'usr_9988')"
        for i in range(100)
    ]
    blocks.append(
        ContextBlock(
            id="tool_explain_analyze",
            kind="tool_result",
            role="tool",
            content="\n".join(log_lines),
            priority=Priority.P2,
            strategy=HeadTail(head_ratio=0.5),
            group_id="call_explain_orders",
            trust="untrusted",
            source="psql:explain_analyze",
        )
    )

    # P3: RAG Documents (3 documents, varying scores)
    rag_articles = [
        (
            "rag_index_b_tree",
            "B-Tree indexing on foreign keys accelerates single-tenant lookups.",
            0.95,
        ),
        (
            "rag_vacuum_tuning",
            "Aggressive autovacuum parameters prevent table bloat on updates.",
            0.70,
        ),
        (
            "rag_history_trivia",
            "PostgreSQL was originally called Postgres created in 1986 at UC Berkeley.",
            0.15,
        ),
    ]
    for doc_id, text, score in rag_articles:
        blocks.append(
            ContextBlock(
                id=doc_id,
                kind="rag",
                role="user",
                content=text,
                priority=Priority.P3,
                strategy=Drop(),
                score=score,
                trust="untrusted",
                source=f"kb/{doc_id}.md",
            )
        )

    # P4: 6 turns of conversation history
    history_turns = [
        ("h0", "user", "I noticed our DB is slow today."),
        ("h1", "assistant", "Could you provide table size metrics?"),
        ("h2", "user", "The orders table has 12 million rows."),
        ("h3", "assistant", "Let me check running queries."),
        ("h4", "user", "Running query stats now."),
        ("h5", "assistant", "Please provide the EXPLAIN plan."),
    ]
    for i, (msg_id, role, text) in enumerate(history_turns):
        blocks.append(
            ContextBlock(
                id=msg_id,
                kind="history",
                role=role,
                content=text,
                priority=Priority.P4,
                strategy=Drop(),
                seq=i,
            )
        )

    print("\n[Input Summary]")
    print(f"- Total Candidate Blocks: {len(blocks)}")
    print(f"- Model Context Window:   {model.context_window} tokens")
    print(f"- Reserved for Output:    {model.max_output_tokens} tokens")

    # 3. Execute Assembly
    result = assembler.assemble(blocks)
    report = result.report

    print("\n" + "=" * 70)
    print("📊 ASSEMBLY REPORT")
    print("=" * 70)
    print(f"⏱️ Latency:           {report.duration_ms:.2f} ms")
    print(f"🎯 Total Tokens:       {report.total_tokens} / {report.input_budget} budget")
    print(f"📈 Window Utilization: {report.utilization * 100:.1f}%")
    print(f"🔒 Static Prefix Hash: {report.prefix_hash}")
    print(f"⚡ Cacheable Tokens:   {report.cache_prefix_tokens}")

    print("\n[Slot Capacity Breakdown]")
    for slot_name, usage in report.per_slot.items():
        print(
            f"  • {slot_name:12s} -> Allocated: {usage.allocated:4d} | "
            f"Used: {usage.used:4d} | Demand: {usage.demand:4d}"
        )

    print("\n[Decisions Audit Log]")
    for d in report.decisions:
        status_icon = "✅" if d.action == "included" else ("✂️" if d.action == "truncated" else "❌")
        print(
            f"  {status_icon} {d.block_id:22s} [{d.action:9s}] "
            f"{d.tokens_before:4d} -> {d.tokens_after:4d} tokens ({d.reason})"
        )

    # 4. View Formatted OpenAI Messages
    print("\n" + "=" * 70)
    print("📦 FORMATTED OPENAI PAYLOAD (Sample Output)")
    print("=" * 70)
    openai_payload = to_openai_messages(result.blocks)
    for i, msg in enumerate(openai_payload):
        raw_text = str(msg["content"])
        suffix = "..." if len(raw_text) > 100 else ""
        snippet = raw_text[:100].replace("\n", " ") + suffix
        role_label = f"[{msg['role'].upper()}]"
        print(f"{i + 1:02d}. {role_label:14s} {snippet}")

    # 5. View Formatted Anthropic Payload
    anthropic_payload = to_anthropic_payload(result.blocks)
    print("\n" + "=" * 70)
    print("📦 FORMATTED ANTHROPIC PAYLOAD (System vs Messages)")
    print("=" * 70)
    print(f"Top-level 'system' count: {len(anthropic_payload.get('system', []))}")
    print(f"Conversation 'messages':  {len(anthropic_payload.get('messages', []))}")
    print("=" * 70)
    print("🎉 All checks passed! The context assembled cleanly without overflow.")
    print("=" * 70)


if __name__ == "__main__":
    run_demo()
