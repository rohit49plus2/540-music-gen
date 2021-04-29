import os.path
from glob import glob
import pickle
from music21 import converter, instrument, note, chord, stream
from music21 import corpus
import keras
import numpy as np
from keras.utils import np_utils
import play

sequence_length = 16

dir_path = os.path.dirname(os.path.realpath(__file__))

songs = glob('data/*.mid')
"Midi Files used:"
print(songs)
print('\n\n')

# chorales = corpus.getBachChorales()
# score  = corpus.parse(chorales[0])

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
            #1 for Format 0 midi files like mond_1.mid and 0 for Format 1 midi files
            notes_to_parse = parts.parts[1].recurse()
        else:
            notes_to_parse = midi.flat.notes
        for element in notes_to_parse:
            # print(element)
            if isinstance(element, note.Note):
                # if element is a note, extract pitch
                notes.append(str(element.pitch))
            elif(isinstance(element, chord.Chord)):
                # if element is a chord, append the normal form of the
                # chord (a list of integers) to the list of notes.
                notes.append('.'.join(str(n) for n in element.normalOrder))
    print("notes",notes)
    with open('data/notes', 'wb') as filepath:
        pickle.dump(notes, filepath)
    return notes


def prepare_sequences(notes, n_vocab):

    # Extract the unique pitches in the list of notes.
    pitchnames = sorted(set(item for item in notes))
    # create a dictionary to map pitches to integers
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))
    int_to_note = dict((number, note) for number, note in enumerate(pitchnames))

    network_inputs = []
    network_outputs = []

    for j in range(1,sequence_length+1):
        network_input = []
        network_output = []

        # create input sequences and the corresponding outputs
        for i in range(0, len(notes) - sequence_length, 1):
            sequence_in = notes[i: i + j]
            sequence_out = notes[i + j]
            network_input.append([note_to_int[char] for char in sequence_in])
            network_output.append(note_to_int[sequence_out])

        n_patterns = len(network_input)

        # reshape the input into a format comatible with LSTM layers
        network_input = np.reshape(network_input, (n_patterns, j, 1))

        # normalize input
        network_input = network_input / float(n_vocab)

        # one hot encode the output vectors
        network_output = np_utils.to_categorical(network_output)

        network_inputs.append(network_input)
        network_outputs.append(network_output)

    return (network_inputs, network_outputs)



from keras.models import Sequential
from keras.layers import Activation, Dense, LSTM, Dropout, Flatten
def create_network(network_inputs, n_vocab):
    models=[]
    for j in range(sequence_length):
        """Create the model architecture"""
        model = Sequential()
        model.add(LSTM(128, input_shape=network_inputs[j].shape[1:], return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(128, return_sequences=True))
        model.add(Flatten())
        model.add(Dense(256))
        model.add(Dropout(0.3))
        model.add(Dense(n_vocab))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        models.append(model)
    return models

from keras.callbacks import ModelCheckpoint
def train(models, network_inputs, network_outputs, epochs):
    """
    Train the neural network
    """
    for j in range(sequence_length):
        network_input=network_inputs[j]
        network_output=network_outputs[j]
        # Create checkpoint to save the best model weights.
        filepath = 'data/weights-GAN/weights.best.music'+str(j)+'.hdf5'
        checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=0, save_best_only=True)

        models[j].fit(network_input, network_output, epochs=epochs, batch_size=32, callbacks=[checkpoint])

def train_network(network_inputs, network_outputs):
    """
    Get notes
    Generates input and output sequences
    Creates a model
    Trains the model for the given epochs
    """

    epochs = 100

    notes = get_notes()
    print('Notes processed')

    n_vocab = len(set(notes))
    print('Vocab generated')

    models = create_network(network_inputs, n_vocab)
    print('Model created')
    print('Training in progress')
    train(models, network_inputs, network_outputs, epochs)
    print('Training completed')
    return models

# training = True
training = False
you_have_processed_notes = True
# you_have_processed_notes = False
def generate():
    """ Generate a piano midi file """
    print('Loading Notes')
    if you_have_processed_notes:
        assert(os.path.exists('data/notes'))
        print("Looks like you've already processed notes, i'll use them")
        #load the notes used to train the model
        with open('data/notes', 'rb') as filepath:
            notes = pickle.load(filepath)
    else:
        print("Extracting notes from midi files")
        notes = get_notes()

    # Get all pitch names
    pitchnames = sorted(set(item for item in notes))
    # Get all pitch names
    n_vocab = len(set(notes))


    network_inputs, network_outputs = prepare_sequences(notes, n_vocab)
    print('Input and Output processed')

    if training:
        models= train_network(network_inputs, network_outputs)
        pass
    else:
        models = create_network(network_inputs,n_vocab)
        print('Loading Model weights.....')
        for j in range(sequence_length):
            models[j].load_weights('data/weights-GAN/weights.best.music'+str(j)+'.hdf5')
        print('Model Loaded')
    print("Creating Output")
    prediction_output = generate_notes(models, network_inputs, pitchnames, n_vocab)
    # for i in range(sequence_length):
    #     print(network_inputs[i].shape)
    #     print(network_outputs[i].shape)
    print("Creating Midi File")
    create_midi(prediction_output)

def generate_notes(models, network_inputs, pitchnames, n_vocab):
    """ Generate notes from the neural network based on a sequence of notes """
    int_to_note = dict((number, note) for number, note in enumerate(pitchnames))

    start_seq = 0
    pattern=[]
    network_input_0=network_inputs[start_seq]*float(n_vocab)
    prediction_output = []

    print('Generating notes........')

    # generate n * sequence_length notes
    n = 10
    for note_index in range(start_seq,n*sequence_length):
        # prediction = np.reshape(np.zeros(n_vocab), (1,n_vocab))
        i = note_index%sequence_length
        if i ==0:
            if len(pattern)>0:
                pattern = pattern[-1:]
            else:
                # pick a random sequence from the input as a starting point for the prediction
                start = np.random.randint(0, len(network_input_0)-1)
                pattern = network_input_0[start].tolist()[0]
        model = models[i]
        prediction_input = np.reshape(pattern, (1, len(pattern), 1))
        prediction_input = np.asarray(prediction_input).astype('float32')
        prediction = model.predict(prediction_input, verbose=0)

        # Predicted output is the argmax(P(h|D))
        index = np.argmax(prediction)

        # Mapping the predicted interger back to the corresponding note
        result = int_to_note[index]
        # Storing the predicted output
        prediction_output.append(result)

        # Next input to the model
        pattern.append(index)
        print(note_index,pattern)

    print('Notes Generated...', prediction_output)
    return prediction_output

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

    midi_stream.write('midi', fp='test_output5.mid')

generate()
