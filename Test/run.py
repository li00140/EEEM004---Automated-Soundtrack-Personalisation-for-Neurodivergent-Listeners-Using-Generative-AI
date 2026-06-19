from Stem_Personalisation import personalise_mix, PRESETS

result = personalise_mix(
    dialogue_path="dialogue.wav",
    music_path="music.wav",
    sfx_path="sfx.wav",
    profile=PRESETS["default"],
    output_path="accessible_mix.wav"
    )

print(result)
