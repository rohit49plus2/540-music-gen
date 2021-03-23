import os.path
from glob import glob
import pickle
from constants_load_music import *
from music21 import converter, instrument, note, chord, stream

#dir_path = dir_path

songs = glob(midi_dir)
print(songs)

def get_notes():
    notes = []
    for file in songs:
     # converting .mid file to stream object
     midi = converter.parse(file)
     notes_to_parse = []
     try:
       # Given a single stream, partition into a part for each unique instrument
       parts = instrument.partitionByInstrument(midi)
     except:
       pass
     if parts: # if parts has instrument parts
       notes_to_parse = parts.parts[0].recurse()
     else:
       notes_to_parse = midi.flat.notes
     for element in notes_to_parse:
       if isinstance(element, note.Note):
         # if element is a note, extract pitch
         notes.append(str(element.pitch))
       elif(isinstance(element, chord.Chord)):
         # if element is a chord, append the normal form of the
         # chord (a list of integers) to the list of notes.
         notes.append('.'.join(str(n) for n in element.normalOrder))
    with open(notes_dir, 'wb') as filepath:
     pickle.dump(notes, filepath)
    return notes

notes=get_notes()
print(notes)

# Extract the unique pitches in the list of notes.
pitchnames = sorted(set(item for item in notes))
# create a dictionary to map pitches to integers
note_to_int = dict((note, number) for number, note in enumerate(pitchnames))
int_to_note = dict((number, note) for number, note in enumerate(pitchnames))

print(int_to_note)
