"""
Local inference — chat with your fine-tuned model in the terminal.
Supports optional RAG context injection for grounded answers.
"""

from pathlib import Path
from typing import Optional


DEFAULT_SYSTEM = (
    "You are a personal AI assistant trained on the user's own documents, "
    "notes, and browsing history. Answer questions drawing on that personal knowledge. "
    "If you don't know something, say so clearly."
)

RAG_SYSTEM = (
    "You are a personal AI assistant. Use the CONTEXT below to answer the user's "
    "question accurately. Only use information from the context. If the context "
    "doesn't contain the answer, say you don't have that information.\n\n"
    "CONTEXT:\n{context}"
)


def load_model(model_dir: Path):
    """Load model and tokenizer from a saved LoRA checkpoint."""
    try:
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            str(model_dir), max_seq_length=2048, dtype=None, load_in_4bit=True
        )
        FastLanguageModel.for_inference(model)
        return model, tokenizer, "unsloth"
    except ImportError:
        pass

    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    import torch

    # Try to find base model name from config
    config_path = model_dir / "adapter_config.json"
    base_model = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    if config_path.exists():
        import json
        cfg = json.loads(config_path.read_text())
        base_model = cfg.get("base_model_name_or_path", base_model)

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    base = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.float16, device_map="auto"
    )
    model = PeftModel.from_pretrained(base, str(model_dir))
    model.eval()
    return model, tokenizer, "transformers"


def generate(model, tokenizer, messages: list[dict], backend: str,
             max_new_tokens: int = 512, temperature: float = 0.7,
             retriever=None) -> tuple[str, list[dict] | None]:
    """
    Generate a response given a message history.
    If retriever is provided, injects relevant context into the system prompt.
    Returns (response_text, retrieved_chunks_or_None).
    """
    import torch

    retrieved = None
    working_messages = list(messages)

    # RAG: retrieve context for the latest user message
    if retriever is not None:
        user_msg = next(
            (m["content"] for m in reversed(working_messages) if m["role"] == "user"),
            None,
        )
        if user_msg:
            from pipeline.rag import format_context
            results = retriever.search(user_msg, top_k=5)
            if results:
                retrieved = results
                context_text = format_context(results)
                rag_system = RAG_SYSTEM.format(context=context_text)
                # Replace system message with RAG-augmented one
                working_messages = [
                    m if m["role"] != "system" else {"role": "system", "content": rag_system}
                    for m in working_messages
                ]

    # Use chat template if available
    try:
        prompt = tokenizer.apply_chat_template(
            working_messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        # Manual fallback
        prompt = "\n".join(
            f"{'User' if m['role']=='user' else 'Assistant'}: {m['content']}"
            for m in working_messages
        ) + "\nAssistant:"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
        )

    decoded = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:],
                               skip_special_tokens=True)
    return decoded.strip(), retrieved


def chat_loop(model_dir: Path, system_prompt: Optional[str] = None,
              use_rag: bool = False):
    """Interactive terminal chat loop."""
    print(f"Loading model from {model_dir}...")
    model, tokenizer, backend = load_model(model_dir)
    print(f"Model loaded ({backend} backend)")

    retriever = None
    if use_rag:
        from pipeline.rag import index_exists, Retriever
        if index_exists():
            retriever = Retriever()
            print("RAG enabled — answers will use your documents as context")
        else:
            print("No RAG index found. Build one with: intelope index --dataset <name>")
    print()

    system = system_prompt or DEFAULT_SYSTEM
    history = [{"role": "system", "content": system}]

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user_input:
            continue
        if user_input == "/exit":
            print("Goodbye.")
            break
        if user_input == "/clear":
            history = [{"role": "system", "content": system}]
            print("[Context cleared]\n")
            continue

        history.append({"role": "user", "content": user_input})
        response, retrieved = generate(model, tokenizer, history, backend,
                                       retriever=retriever)
        history.append({"role": "assistant", "content": response})

        if retrieved:
            sources = set(r["source"] for r in retrieved if r["source"])
            if sources:
                print(f"\n  [Sources: {', '.join(sources)}]")
        print(f"\nAssistant: {response}\n")
