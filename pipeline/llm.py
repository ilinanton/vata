"""Step 3: LLM integration (Ollama / OpenRouter) — speaker naming and analytics."""

import json


def create_client(config: dict, env: dict) -> tuple:
    """Create an LLM client based on the provider from config.

    Returns (client, provider_name).
    """
    provider = config["llm"]["provider"]

    if provider == "ollama":
        import ollama

        base_url = config["llm"]["ollama"].get("base_url", "http://localhost:11434")
        client = ollama.Client(host=base_url)
        return client, "ollama"

    elif provider == "openrouter":
        from openai import OpenAI

        api_key = env.get("OPENROUTER_API_KEY", "")
        base_url = config["llm"]["openrouter"].get("base_url", "https://openrouter.ai/api/v1")
        client = OpenAI(api_key=api_key, base_url=base_url)
        return client, "openrouter"

    else:
        raise ValueError(f"Unknown LLM provider: {provider}")


def _build_transcript_text(segments: list[dict], max_chars: int = 8000) -> str:
    """Build transcript text for the prompt."""
    lines = []
    total = 0
    for seg in segments:
        line = f"{seg['speaker']}: {seg['text']}"
        if total + len(line) > max_chars:
            lines.append("...")
            break
        lines.append(line)
        total += len(line)
    return "\n".join(lines)


def _chat(client, model: str, system: str, user: str, provider: str) -> str:
    """Universal chat call for Ollama and OpenRouter."""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    if provider == "ollama":
        response = client.chat(model=model, messages=messages)
        return response["message"]["content"]
    else:
        response = client.chat.completions.create(model=model, messages=messages)
        return response.choices[0].message.content


def _parse_json(text: str) -> dict | None:
    """Extract JSON from LLM response."""
    # Try to find JSON in ```json...``` block
    if "```json" in text:
        start = text.index("```json") + 7
        end = text.index("```", start)
        text = text[start:end].strip()
    elif "```" in text:
        start = text.index("```") + 3
        end = text.index("```", start)
        text = text[start:end].strip()

    # Try to find JSON object
    for i, ch in enumerate(text):
        if ch == "{":
            text = text[i:]
            break

    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None


def name_speakers(client, model: str, segments: list[dict]) -> dict[str, str]:
    """Identify speaker names via LLM.

    Returns a mapping: {"SPEAKER_00": "Alex", "SPEAKER_01": "Maria", ...}
    """
    speakers = sorted(set(seg["speaker"] for seg in segments))
    if not speakers:
        return {}

    transcript = _build_transcript_text(segments)

    # Determine provider by client type
    provider = "ollama" if hasattr(client, "chat") and not hasattr(client, "completions") else "openrouter"

    system = (
        "You are an assistant for analyzing transcriptions. "
        "Identify the likely names of speakers based on the conversation context. "
        "If a name cannot be determined, create an appropriate label (e.g., 'Host', 'Guest')."
    )
    user = (
        f"Here is a conversation transcript with {len(speakers)} speakers:\n\n"
        f"{transcript}\n\n"
        f"Speakers: {', '.join(speakers)}\n\n"
        "Return a JSON object mapping speaker identifiers to their likely names. "
        "Format: {\"SPEAKER_00\": \"Name\", \"SPEAKER_01\": \"Name\"}\n"
        "Return ONLY JSON, no explanations."
    )

    try:
        response = _chat(client, model, system, user, provider)
        mapping = _parse_json(response)
        if mapping and isinstance(mapping, dict):
            # Ensure all speakers are in the mapping
            for sp in speakers:
                if sp not in mapping:
                    mapping[sp] = sp
            return mapping
    except Exception:
        pass

    # Fallback: original names
    return {sp: sp for sp in speakers}


def generate_analytics(
    client, model: str, segments: list[dict], source_file: str
) -> dict:
    """Generate analytics: title, date, summary.

    Returns {"title": ..., "date": ..., "summary": ...}
    """
    transcript = _build_transcript_text(segments)

    provider = "ollama" if hasattr(client, "chat") and not hasattr(client, "completions") else "openrouter"

    system = (
        "You are an assistant for analyzing meeting and interview transcriptions. "
        "Generate metadata based on the conversation content."
    )
    user = (
        f"Here is a transcript from file '{source_file}':\n\n"
        f"{transcript}\n\n"
        "Generate:\n"
        "1. title — a brief conversation title\n"
        "2. date — the conversation date (YYYY-MM-DD) if mentioned; otherwise an empty string\n"
        "3. summary — a brief summary in 3-5 sentences\n\n"
        "Return JSON: {\"title\": \"...\", \"date\": \"...\", \"summary\": \"...\"}\n"
        "Return ONLY JSON, no explanations."
    )

    try:
        response = _chat(client, model, system, user, provider)
        result = _parse_json(response)
        if result and isinstance(result, dict):
            return {
                "title": result.get("title", source_file),
                "date": result.get("date", ""),
                "summary": result.get("summary", ""),
            }
    except Exception:
        pass

    # Fallback
    return {"title": source_file, "date": "", "summary": ""}


def apply_speaker_names(segments: list[dict], mapping: dict[str, str]) -> list[dict]:
    """Replace speaker identifiers with readable names."""
    return [
        {**seg, "speaker": mapping.get(seg["speaker"], seg["speaker"])}
        for seg in segments
    ]
