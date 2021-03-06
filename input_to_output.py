# takes midi file and extracts notes, then converts notes into midi

import os.path
from glob import glob
import pickle
from music21 import converter, instrument, note, chord, stream
import keras
import numpy as np
from keras.utils import np_utils
import play


dir_path = os.path.dirname(os.path.realpath(__file__))

songs = glob(dir_path+'/data/*.mid')
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
       print(element)
       if isinstance(element, note.Note):
         # if element is a note, extract pitch
         notes.append(str(element.pitch))
       elif(isinstance(element, chord.Chord)):
         # if element is a chord, append the normal form of the
         # chord (a list of integers) to the list of notes.
         notes.append('.'.join(str(n) for n in element.normalOrder))
    with open(dir_path+'/data/notes', 'wb') as filepath:
     pickle.dump(notes, filepath)
    return notes

notes = get_notes()[:100]

def create_midi(prediction_output):
    """ convert the output from the prediction to notes and create a midi file
        from the notes """
    offset = 0
    output_notes = []

    # create note and chord objects based on the values generated by the model
    for pattern in prediction_output:
        # pattern is a chord
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        # pattern is a note
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)

        # increase offset each iteration so that notes do not stack
        offset += 0.5

    midi_stream = stream.Stream(output_notes)

    print('Saving Output file as midi....')

    midi_stream.write('midi', fp=dir_path+'/test_output2.mid')


create_midi(notes) #see what the network receives as an input
