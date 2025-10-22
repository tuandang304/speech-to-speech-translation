import json
import os

HISTORY_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'results', 'history.json'))

def save_log(log_entry):
    """
    Appends a new log entry to the history.json file.
    """
    history = []
    # Ensure the directory exists
    os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)

    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                history = json.load(f)
        except json.JSONDecodeError:
            # If file is corrupted or empty, start fresh
            history = []

    history.append(log_entry)

    with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=4, ensure_ascii=False)