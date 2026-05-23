import pyttsx3
import sys

text = sys.argv[1]
outfile = sys.argv[2]
engine = pyttsx3.init()
engine.save_to_file(text, outfile)
engine.runAndWait()