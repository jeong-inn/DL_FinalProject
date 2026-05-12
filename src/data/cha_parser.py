from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class Utterance:
    order: int
    speaker: str
    text: str
    clean_text: str
    start_ms: Optional[int] = field(default=None)
    end_ms: Optional[int] = field(default=None)


_TIMESTAMP_RE = re.compile(r'\x15(\d+)_(\d+)\x15')
_RETRACE_RE = re.compile(r'<[^>]+>\s*\[//?\]')
_BRACKET_RE = re.compile(r'\[[^\]]*\]')
_SPECIAL_RE = re.compile(r'[&+<>]')
_MULTI_SPACE_RE = re.compile(r'\s+')


def _extract_timestamp(text: str) -> tuple[Optional[int], Optional[int], str]:
    m = _TIMESTAMP_RE.search(text)
    if m:
        return int(m.group(1)), int(m.group(2)), text[:m.start()] + text[m.end():]
    return None, None, text


def clean(text: str) -> str:
    text = _RETRACE_RE.sub('', text)
    text = _BRACKET_RE.sub('', text)
    text = _TIMESTAMP_RE.sub('', text)
    text = _SPECIAL_RE.sub('', text)
    text = text.replace('xxx', '').replace('yyy', '').replace('www', '')
    return _MULTI_SPACE_RE.sub(' ', text).strip()


def extract_utterances(filepath: str | Path, speakers: Optional[list[str]] = None) -> list[Utterance]:
    utterances = []
    order = 0

    with open(filepath, encoding='utf-8', errors='replace') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith('*') and ':' in line:
            colon = line.index(':')
            speaker = line[1:colon]
            text = line[colon + 1:].strip()

            j = i + 1
            while j < len(lines) and lines[j].startswith('\t') and not lines[j][1:].startswith('%'):
                text += ' ' + lines[j].strip()
                j += 1

            start_ms, end_ms, text_no_ts = _extract_timestamp(text)

            if speakers is None or speaker in speakers:
                order += 1
                utterances.append(Utterance(
                    order=order,
                    speaker=speaker,
                    text=text_no_ts.strip(),
                    clean_text=clean(text),
                    start_ms=start_ms,
                    end_ms=end_ms,
                ))
            i = j
        else:
            i += 1

    return utterances


def count_utterance_by_speaker(filepath: str | Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for utt in extract_utterances(filepath):
        counts[utt.speaker] = counts.get(utt.speaker, 0) + 1
    return counts


def get_child_text(filepath: str | Path) -> str:
    utts = extract_utterances(filepath, speakers=['CHI'])
    return ' '.join(u.clean_text for u in utts if u.clean_text)


def has_timestamps(filepath: str | Path) -> bool:
    """Return True if the .cha file contains audio timestamp markers."""
    utts = extract_utterances(filepath)
    return any(u.start_ms is not None for u in utts)
