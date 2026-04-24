import argparse
import re
from pathlib import Path
from typing import Any, Dict

from .tracker import PROJECT_ROOT, resolve_project_path, track_video_and_center_object


def get_next_run_name(base_name: str, runs_relative_path: str) -> str:
    runs_dir = resolve_project_path(runs_relative_path)
    runs_dir.mkdir(parents=True, exist_ok=True)

    pattern = re.compile(rf"^{re.escape(base_name)}_v(\d+)$")
    max_version = 0
    for item in runs_dir.iterdir():
        if not item.is_dir():
            continue
        match = pattern.match(item.name)
        if match:
            max_version = max(max_version, int(match.group(1)))

    return f"{base_name}_v{max_version + 1}"


def load_config(config_path: Path) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML не установлен. Установите зависимости: pip install -r requirements.txt") from exc

    config_path = resolve_project_path(config_path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Файл конфигурации не найден: '{config_path}'")

    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(f"Файл конфигурации должен содержать YAML-словарь: '{config_path}'")
    if "paths" not in config or "tracking" not in config:
        raise ValueError("В config.yaml должны быть секции 'paths' и 'tracking'.")

    return config


def build_tracking_paths(config: Dict[str, Any]) -> Dict[str, Path]:
    paths = config["paths"]
    track_run_name = get_next_run_name("snowboarder_tracking", runs_relative_path="runs/track")
    video_output_template = paths["video_output_path"].format(track_run_name=track_run_name)

    return {
        "model_path": resolve_project_path(paths["model_path"]),
        "video_input_path": resolve_project_path(paths["video_input_path"]),
        "video_output_path": resolve_project_path(video_output_template),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run snowboarder tracking from config.yaml.")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "config.yaml"),
        help="Path to config.yaml. Relative paths are resolved from the project root.",
    )
    args = parser.parse_args()

    config = load_config(Path(args.config))
    paths = build_tracking_paths(config)

    track_video_and_center_object(
        model_path=paths["model_path"],
        video_input_path=paths["video_input_path"],
        video_output_path=paths["video_output_path"],
        config=config,
    )


if __name__ == "__main__":
    main()
