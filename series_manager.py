"""
series_manager.py - Multi-Part Sermon Series Engine for Pastor Tara AI Assistant

Detects sermon series, groups multi-part messages (e.g. Part 1/2/3, Part A/B/C),
supplements missing parts from the ministry website archives (sermons.jsonl),
and ensures that when any part of a message is delivered, ALL parts are provided
in proper chronological order.
"""

import json
import logging
import os
import re
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHANNEL_MESSAGES_PATH = os.path.join(BASE_DIR, "utilities", "channel_messages.jsonl")
COMBINED_MESSAGES_PATH = os.path.join(BASE_DIR, "utilities", "combined_channel_messages.jsonl")
SERMONS_PATH = os.path.join(BASE_DIR, "utilities", "sermons.jsonl")


def normalize_series_key(s: str) -> str:
    """Normalize a title for robust fuzzy/series matching."""
    s = s.lower().replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    s = re.sub(r"\.(mp3|m4a|wav|aac|ogg)$", "", s, flags=re.I)
    s = s.replace("-", " ")
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    # Normalize common singular/plural variations
    s = re.sub(r"\bprayers\b", "prayer", s)
    s = re.sub(r"\bmiracles\b", "miracle", s)
    s = re.sub(r"\bencounters\b", "encounter", s)
    return s


def parse_series_info(raw_title: str) -> tuple[str, str, int]:
    """
    Extract base title, part label, and numeric order from a sermon title.
    
    Examples:
      'How to Plant Greatness B' -> ('How to Plant Greatness', 'Part B', 2)
      'Christ, the Wisdom of God 1' -> ('Christ, the Wisdom of God', 'Part 1', 1)
      'Experiencing Favour 3' -> ('Experiencing Favour', 'Part 3', 3)
      'The Praise of the Righteous-A' -> ('The Praise of the Righteous', 'Part A', 1)
      'Enforcing Victory' -> ('Enforcing Victory', 'Part 1', 1)
    """
    title = re.sub(r"\.(mp3|m4a|wav|aac|ogg)$", "", raw_title, flags=re.I).replace("-", " ")
    title = re.sub(r"\s+", " ", title).strip()

    # Pattern 1: Title ending with Part X or Pt X or space + [A-E|1-9]
    # e.g., 'How to Plant Greatness B', 'The Power of God 2', 'Faith for Life Part 1'
    m = re.search(r"^(.*?)(?:\s+[-–—,]?\s*(?:part|pt|episode|ep)?\s*([a-e]|\d+))\s*$", title, re.I)
    if m:
        base = m.group(1).strip()
        part_token = m.group(2).strip().upper()
        if part_token.isdigit():
            order = int(part_token)
            label = f"Part {order}"
        else:
            order = ord(part_token) - ord("A") + 1
            label = f"Part {part_token}"
        return base, label, order

    return title, None, 1


def get_sermon_title(msg: dict) -> str:
    """Extract human-readable title from message metadata, filename, or text."""
    if not msg:
        return "Prophetic Teaching"
    if msg.get("title"):
        return msg["title"].strip()
    if msg.get("audio_filename"):
        fn = msg["audio_filename"]
        clean = re.sub(r"\.(mp3|m4a|wav|aac|ogg)$", "", fn, flags=re.I)
        clean = clean.replace("-", " ").strip()
        clean = re.sub(r"\s+", " ", clean)
        if len(clean) > 2:
            return clean
    text = msg.get("text", "")
    match = re.search(r"[“\"']([^“”\"'\n]{3,60})[”\"']", text)
    if match:
        return match.group(1).strip()
    return "Prophetic Teaching"


class SeriesEngine:
    def __init__(self):
        # Maps normalized_key -> list of part dicts
        self.series_by_key: dict[str, list[dict]] = defaultdict(list)
        # Maps message_id -> series list
        self.msg_id_to_series: dict[int, list[dict]] = {}
        # Maps normalized_title -> series list
        self.title_to_series: dict[str, list[dict]] = {}
        self.is_loaded = False

    def build_index(self, channel_messages: list, website_sermons: list = None):
        """Build the full series index from channel posts and website archives."""
        self.series_by_key.clear()
        self.msg_id_to_series.clear()
        self.title_to_series.clear()

        # Step 1: Index all channel messages
        for msg in channel_messages:
            title = get_sermon_title(msg)
            base_title, part_label, order = parse_series_info(title)
            norm_key = normalize_series_key(base_title)

            entry = {
                "source": "channel",
                "base_title": base_title,
                "display_title": self._format_display_title(base_title, part_label),
                "part_label": part_label,
                "order": order,
                "msg": msg,
            }
            self.series_by_key[norm_key].append(entry)

        # Step 2: Supplement missing parts from website sermons (sermons.jsonl)
        if website_sermons:
            for s in website_sermons:
                s_title = s.get("title", "")
                s_audio = s.get("audio_url", "")
                fn = s_audio.split("/")[-1].replace(".mp3", "") if s_audio else s_title
                base_title, part_label, order = parse_series_info(fn)
                if not base_title or base_title.lower() == "prophetic teaching":
                    base_title, part_label, order = parse_series_info(s_title)

                norm_key = normalize_series_key(base_title)
                # Check if this base matches an existing series in channel
                if norm_key in self.series_by_key:
                    existing = self.series_by_key[norm_key]
                    existing_orders = [x["order"] for x in existing]
                    if order not in existing_orders:
                        # Construct a compatible message object for the website audio
                        site_msg = {
                            "message_id": None,
                            "text": s.get("description", "") or s.get("title", ""),
                            "audio_filename": fn + ".mp3",
                            "audio_url": s_audio,
                            "channel_link": s.get("url", ""),
                            "title": self._format_display_title(base_title, part_label),
                        }
                        self.series_by_key[norm_key].append({
                            "source": "website",
                            "base_title": base_title,
                            "display_title": site_msg["title"],
                            "part_label": part_label,
                            "order": order,
                            "msg": site_msg,
                        })

        # Step 3: Sort each series chronologically and build lookup tables
        for norm_key, items in self.series_by_key.items():
            # If any item has an explicit part label, treat all items in this group as a series
            has_parts = any(x["part_label"] is not None for x in items)
            if len(items) > 1 or has_parts:
                # Assign default Part 1 label to Base if other parts exist
                for x in items:
                    if x["part_label"] is None and len(items) > 1:
                        x["part_label"] = "Part 1"
                        x["display_title"] = self._format_display_title(x["base_title"], "Part 1")

                items.sort(key=lambda x: (x["order"], 0 if x["part_label"] else 1, x["msg"].get("message_id") or 999999))
                # Deduplicate by order so we don't have multiple 'Part 1's for the same series
                unique_parts = []
                seen_orders = set()
                for x in items:
                    if x["order"] not in seen_orders:
                        seen_orders.add(x["order"])
                        unique_parts.append(x)
                items[:] = unique_parts

            # Populate fast lookup dictionaries
            for x in items:
                mid = x["msg"].get("message_id")
                if mid:
                    self.msg_id_to_series[mid] = items
                norm_full = normalize_series_key(get_sermon_title(x["msg"]))
                self.title_to_series[norm_full] = items

        self.is_loaded = True
        multi_count = sum(1 for v in self.series_by_key.values() if len(v) > 1)
        logging.info(
            f"SeriesEngine initialized: {len(self.series_by_key)} series keys, "
            f"{multi_count} multi-part series ready."
        )

    def _format_display_title(self, base: str, part_label: str) -> str:
        if part_label:
            return f"{base} — {part_label}"
        return base

    def get_series_for_message(self, msg: dict) -> list[dict]:
        """
        Return all parts of the series that this message belongs to,
        in sequential order (Part 1, Part 2, Part 3, etc.).
        If the message is standalone, returns a list containing just this message's entry.
        """
        if not self.is_loaded:
            return [{"msg": msg, "display_title": get_sermon_title(msg), "order": 1, "part_label": None}]

        mid = msg.get("message_id")
        if mid and mid in self.msg_id_to_series:
            series = self.msg_id_to_series[mid]
            if len(series) > 1 or any(x.get("part_label") for x in series):
                return series

        title = get_sermon_title(msg)
        norm_title = normalize_series_key(title)
        if norm_title in self.title_to_series:
            series = self.title_to_series[norm_title]
            if len(series) > 1 or any(x.get("part_label") for x in series):
                return series

        base_title, part_label, order = parse_series_info(title)
        norm_key = normalize_series_key(base_title)
        if norm_key in self.series_by_key:
            series = self.series_by_key[norm_key]
            if len(series) > 1 or any(x.get("part_label") for x in series):
                return series

        return [{
            "source": "channel" if mid else "website",
            "base_title": base_title,
            "display_title": self._format_display_title(base_title, part_label),
            "part_label": part_label,
            "order": order,
            "msg": msg,
        }]


# Global singleton instance
SERIES_ENGINE = SeriesEngine()


def init_series_engine(channel_messages: list, sermons_path: str = SERMONS_PATH):
    """Initialize the series engine from channel messages and sermons archive."""
    site_sermons = []
    if os.path.exists(sermons_path):
        try:
            with open(sermons_path, encoding="utf-8") as f:
                site_sermons = [json.loads(l) for l in f if l.strip()]
        except Exception as e:
            logging.warning(f"Could not load website sermons archive: {e}")

    SERIES_ENGINE.build_index(channel_messages, site_sermons)


def group_ranked_teachings(ranked_list: list) -> list[list[dict]]:
    """
    Groups ranked messages by sermon series.
    Each element in the returned list is a list of sermon parts in sequential order.
    Duplicate mentions of parts from the same series later in the list are deduplicated.
    """
    seen_bases = set()
    grouped = []

    for item in ranked_list:
        msg = item[0] if isinstance(item, (list, tuple)) else item
        series = SERIES_ENGINE.get_series_for_message(msg)

        # Get canonical base key for this teaching
        base_title = series[0].get("base_title") or get_sermon_title(msg)
        norm_base = normalize_series_key(base_title)

        if norm_base in seen_bases:
            continue
        seen_bases.add(norm_base)

        # Extract message dicts from series parts
        teaching_parts = []
        for part in series:
            part_msg = part["msg"].copy()
            part_msg["display_title"] = part.get("display_title") or get_sermon_title(part_msg)
            part_msg["part_label"] = part.get("part_label")
            part_msg["order"] = part.get("order", 1)
            teaching_parts.append(part_msg)

        grouped.append(teaching_parts)

    return grouped

