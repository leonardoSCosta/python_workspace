import gtts

texto = ("""Como ?está se sentindo?""")

tts = gtts.gTTS(text=texto, lang='pt')
tts.save("fala.mp3")
