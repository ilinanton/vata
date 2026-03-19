"""Step 4: render transcript to Markdown via Jinja2."""

from pathlib import Path

from jinja2 import Environment, FileSystemLoader


def render_transcript(data: dict, output_path: Path) -> Path:
    """Render transcript data to a Markdown file.

    Args:
        data: dict with keys title, date, participants, source_file,
              whisper_model, analytics_model, duration, summary, segments.
        output_path: path to save the result.

    Returns:
        Path to the created file.
    """
    templates_dir = Path(__file__).parent.parent / "templates"
    env = Environment(loader=FileSystemLoader(str(templates_dir)))
    template = env.get_template("transcript.md.j2")

    rendered = template.render(**data)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered, encoding="utf-8")

    return output_path
