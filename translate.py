def translate_notes_to_english(notes: list[str]) -> str:
    """
    Translate a list of musical notes to an English description.
    """
    if not notes:
        return "No distinct notes were detected."

    if len(notes) == 1:
        return f"The distinct note detected is {notes[0]}."

    note_list = ", ".join(notes[:-1]) + f", and {notes[-1]}"
    return f"The distinct notes detected are {note_list}."


if __name__ == "__main__":
    # Example usage
    notes = ["C4", "D5", "E6"]
    print(translate_notes_to_english(notes))
