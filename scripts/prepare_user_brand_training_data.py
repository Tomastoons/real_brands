import argparse
import json
from pathlib import Path
import re


URL_SPAN_PATTERN = re.compile(r"\[[^\]\s]*\.[^\]\s]+\]|https?://[^\s)\]>]+|(?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,}")
TOKEN_CHAR_PATTERN = re.compile(r"[A-Za-z0-9&./\-]")


def _find_url_spans(text: str) -> list[tuple[int, int]]:
    return [(m.start(), m.end()) for m in URL_SPAN_PATTERN.finditer(text)]


def _trim_span(text: str, start: int, end: int) -> tuple[int, int]:
    s = max(0, start)
    e = min(len(text), end)
    while s < e and text[s].isspace():
        s += 1
    while e > s and text[e - 1].isspace():
        e -= 1
    while s < e and text[s] in "([{'\"`":
        s += 1
    while e > s and text[e - 1] in ")]}'\"`.,;:":
        e -= 1
    return s, e


def _expand_to_token_boundaries(text: str, start: int, end: int) -> tuple[int, int]:
    s = min(max(0, start), len(text))
    e = min(max(0, end), len(text))
    if e < s:
        e = s
    while s > 0 and TOKEN_CHAR_PATTERN.fullmatch(text[s - 1]):
        s -= 1
    while e < len(text) and TOKEN_CHAR_PATTERN.fullmatch(text[e]):
        e += 1
    return _trim_span(text, s, e)


def _closest_span(target_start: int, spans: list[tuple[int, int]]) -> tuple[int, int] | None:
    if not spans:
        return None
    return min(spans, key=lambda item: abs(item[0] - target_start))


def _is_valid_span_text(span_text: str, label: str) -> bool:
    cleaned = span_text.strip()
    if not cleaned:
        return False
    if not any(ch.isalnum() for ch in cleaned):
        return False
    if label == "URL":
        return "." in cleaned or cleaned.startswith("http")
    return True


def _auto_correct_entities(text: str, entities: list[tuple[int, int, str]]) -> list[tuple[int, int, str]]:
    url_spans = _find_url_spans(text)
    corrected: list[tuple[int, int, str]] = []

    for start, end, label in entities:
        s = int(start)
        e = int(end)

        if label == "URL":
            nearest = _closest_span(s, url_spans)
            if nearest is not None:
                s, e = nearest
            else:
                s, e = _expand_to_token_boundaries(text, s, e)
        else:
            s, e = _expand_to_token_boundaries(text, s, e)

        if s >= e:
            continue
        span_text = text[s:e]
        if not _is_valid_span_text(span_text, label):
            continue
        corrected.append((s, e, label))

    # SpaCy NER cannot train with overlapping spans in one example.
    occupied: list[tuple[int, int]] = []
    deduped: list[tuple[int, int, str]] = []
    for s, e, label in sorted(corrected, key=lambda item: (item[0], item[1])):
        if any(not (e <= os or s >= oe) for os, oe in occupied):
            continue
        occupied.append((s, e))
        deduped.append((s, e, label))

    return deduped


def build_user_training_data() -> list[dict]:
    training_data = [
        ("Spotify [spotify.com]", {"entities": [(0, 6, "ORG"), (7, 18, "PRODUCT"), (19, 29, "URL")]}),
        ("Apple Music [music.apple.com]", {"entities": [(0, 11, "ORG"), (12, 27, "PRODUCT"), (28, 48, "URL")]}),
        ("YouTube Music [music.youtube.com]", {"entities": [(0, 13, "ORG"), (14, 33, "PRODUCT"), (34, 56, "URL")]}),
        ("Amazon Music [music.amazon.com]", {"entities": [(0, 12, "ORG"), (13, 28, "PRODUCT"), (29, 51, "URL")]}),
        ("Tidal [tidal.com]", {"entities": [(0, 5, "ORG"), (6, 14, "PRODUCT")]}),
        ("Deezer [deezer.com]", {"entities": [(0, 6, "ORG"), (7, 15, "PRODUCT")]}),
        ("Qobuz", {"entities": [(0, 5, "PRODUCT")]}),
        ("SoundCloud [soundcloud.com]", {"entities": [(0, 10, "ORG"), (11, 23, "URL")]}),
        ("Gaana", {"entities": [(0, 5, "PRODUCT")]}),
        ("JioSaavn", {"entities": [(0, 8, "PRODUCT")]}),
        ("Wynk Music", {"entities": [(0, 10, "ORG"), (11, 20, "PRODUCT")]}),
        ("Hungama Music", {"entities": [(0, 13, "ORG"), (14, 26, "PRODUCT")]}),
        ("Resso", {"entities": [(0, 4, "PRODUCT")]}),
        ("Idagio", {"entities": [(0, 6, "PRODUCT")]}),
        ("Boosteroid", {"entities": [(0, 10, "ORG"), (11, 20, "PRODUCT")]}),
        ("Shadow", {"entities": [(0, 6, "ORG"), (7, 12, "PRODUCT")]}),
        ("GeForce Now [geforce.com]", {"entities": [(0, 13, "ORG"), (14, 24, "PRODUCT"), (25, 36, "URL")]}),
        ("Xbox Game Pass [microsoft.com]", {"entities": [(0, 4, "ORG"), (5, 15, "PRODUCT"), (16, 31, "URL")]}),
        ("PlayStation Plus [playstation.com]", {"entities": [(0, 8, "ORG"), (9, 22, "PRODUCT"), (23, 38, "URL")]}),
        ("Luna", {"entities": [(0, 4, "PRODUCT")]}),
        ("Nvidia GeForce Now", {"entities": [(0, 19, "ORG"), (20, 37, "PRODUCT")]}),
        ("Xbox Cloud Gaming", {"entities": [(0, 20, "ORG"), (21, 39, "PRODUCT")]}),
        ("Boosteroid", {"entities": [(0, 10, "ORG"), (11, 20, "PRODUCT")]}),
        ("Utomik", {"entities": [(0, 6, "PRODUCT")]}),
        ("Google Maps", {"entities": [(0, 11, "ORG"), (12, 22, "PRODUCT")]}),
        ("WeChat", {"entities": [(0, 6, "ORG"), (7, 13, "PRODUCT")]}),
        ("WhatsApp", {"entities": [(0, 8, "ORG"), (9, 16, "PRODUCT")]}),
        ("Snapchat", {"entities": [(0, 8, "ORG"), (9, 17, "PRODUCT")]}),
        ("Life360", {"entities": [(0, 7, "ORG"), (8, 14, "PRODUCT")]}),
        ("Glympse", {"entities": [(0, 6, "ORG"), (7, 13, "PRODUCT")]}),
        ("Find My", {"entities": [(0, 7, "ORG"), (8, 14, "PRODUCT")]}),
        ("FamiSafe", {"entities": [(0, 8, "PRODUCT")]}),
        ("Google Find My Device", {"entities": [(0, 23, "ORG"), (24, 44, "PRODUCT")]}),
        ("Apple Find My", {"entities": [(0, 13, "ORG"), (14, 24, "PRODUCT")]}),
        ("Google Maps", {"entities": [(0, 11, "ORG"), (12, 22, "PRODUCT")]}),
        ("Life360", {"entities": [(0, 7, "ORG"), (8, 14, "PRODUCT")]}),
        ("GeoZilla", {"entities": [(0, 7, "PRODUCT")]}),
        ("NauNau", {"entities": [(0, 6, "PRODUCT")]}),
        ("GeoZilla", {"entities": [(0, 7, "PRODUCT")]}),
        ("OwnTracks", {"entities": [(0, 9, "PRODUCT")]}),
        ("GPS BodyGuard", {"entities": [(0, 14, "PRODUCT")]}),
        ("Find My Kids", {"entities": [(0, 11, "ORG"), (12, 22, "PRODUCT")]}),
        ("Mapy.cz", {"entities": [(0, 6, "PRODUCT")]}),
        ("Zood Location", {"entities": [(0, 13, "PRODUCT")]}),
        ("Famisafe", {"entities": [(0, 7, "PRODUCT")]}),
        ("Gadgets Now", {"entities": [(0, 11, "ORG"), (12, 22, "PRODUCT")]}),
    ]

    rows: list[dict] = []
    for text, annotation in training_data:
        entities = annotation.get("entities", [])
        corrected = _auto_correct_entities(text, entities)
        rows.append({"text": text, "entities": corrected})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Write user-provided SpaCy training data to JSONL")
    parser.add_argument("--output", default="training/user_brand_data.jsonl", help="Output JSONL file path")
    args = parser.parse_args()

    data = build_user_training_data()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for row in data:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    entity_count = sum(len(row.get("entities", [])) for row in data)
    print(f"Wrote {len(data)} records ({entity_count} entities after auto-correction) to {output_path}")


if __name__ == "__main__":
    main()
