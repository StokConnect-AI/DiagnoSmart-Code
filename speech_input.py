import speech_recognition as sr

# Global mode flag
_input_mode = None

def choose_input_mode():
    global _input_mode
    print("🔧 Choose your input method for this session:")
    mode = input("Type 'speak' to use voice or press Enter to type: ").strip().lower()
    _input_mode = "speak" if mode == "speak" else "type"

def get_input(prompt_text, cast_type=str, fallback=None):
    global _input_mode
    print(prompt_text)

    # If mode not chosen yet, ask once
    if _input_mode is None:
        choose_input_mode()

    # If user chose voice
    if _input_mode == "speak":
        recognizer = sr.Recognizer()
        try:
            with sr.Microphone() as source:
                print("🎙️ Listening...")
                audio = recognizer.listen(source, timeout=5)
                text = recognizer.recognize_google(audio)
                print(f"You said: {text}")
                return cast_type(text.strip())
        except (sr.UnknownValueError, sr.RequestError, sr.WaitTimeoutError):
            print("⚠️ Voice input failed. Switching to typing for the rest of this session.")
            _input_mode = "type"

    # Fallback to typing
    try:
        return cast_type(input("Your input: ").strip())
    except Exception:
        return fallback if fallback is not None else cast_type()
