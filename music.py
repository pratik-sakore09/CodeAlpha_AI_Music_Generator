import os
# Suppress TensorFlow logging and oneDNN operations warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import sys
import streamlit as st

# Prevent executing the file directly via standard Python (which triggers missing ScriptRunContext warnings)
if not st.runtime.exists():
    print("\n" + "="*75)
    print(" ERROR: Streamlit web apps cannot be started using 'python music.py'")
    print("="*75)
    print(" To launch the dashboard web page, please run:")
    print("     streamlit run music.py")
    print("\n Or simply double-click the 'run.bat' file in this folder.")
    print("="*75 + "\n")
    sys.exit(0)

import glob
import random
import numpy as np
import pickle
import base64
import time
import pandas as pd
import matplotlib.pyplot as plt
import keras

from music21 import corpus, converter, instrument, note, chord, stream
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, BatchNormalization
from keras.utils import to_categorical

# Set page configuration
st.set_page_config(
    page_title="AI Music Generator | CodeAlpha",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling for a premium aesthetic
st.markdown("""
<style>
    /* Theme color variables */
    :root {
        --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        --secondary-gradient: linear-gradient(135deg, #a1c4fd 0%, #c2e9fb 100%);
    }
    
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 24px;
        border-radius: 16px;
        color: white;
        text-align: center;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    }
    .main-header h1 {
        margin: 0;
        font-family: 'Outfit', 'Inter', sans-serif;
        font-weight: 800;
        font-size: 2.5rem;
        letter-spacing: -0.5px;
    }
    .main-header p {
        margin: 8px 0 0 0;
        opacity: 0.9;
        font-size: 1.1rem;
    }
    .stats-card {
        background-color: #f8f9fa;
        border-left: 5px solid #667eea;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        margin-bottom: 15px;
    }
    .stats-number {
        font-size: 1.8rem;
        font-weight: bold;
        color: #4a5568;
    }
    .stats-label {
        font-size: 0.9rem;
        color: #718096;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .dark .stats-card {
        background-color: #2d3748;
        border-left: 5px solid #764ba2;
    }
    .dark .stats-number {
        color: #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MIDI_FOLDER = os.path.join(SCRIPT_DIR, "midi_songs")
os.makedirs(MIDI_FOLDER, exist_ok=True)

# ============================================================
# UTILITY FUNCTIONS & CACHING
# ============================================================

@st.cache_data(show_spinner="Parsing MIDI files (cached)...")
def load_and_cache_notes(midi_folder):
    """Parse all MIDI files and extract notes/chords."""
    all_notes = []
    midi_files = glob.glob(os.path.join(midi_folder, "*.mid"))
    parsed_files = []

    if not midi_files:
        return [], []

    for file in midi_files:
        try:
            midi = converter.parse(file)
            # Get notes recursively (compatible across music21 versions)
            try:
                notes_to_parse = midi.recurse().notes
            except AttributeError:
                try:
                    notes_to_parse = midi.flat.notes
                except AttributeError:
                    notes_to_parse = midi.flatten().notes

            file_notes_count = 0
            for element in notes_to_parse:
                if isinstance(element, note.Note):
                    all_notes.append(str(element.pitch))
                    file_notes_count += 1
                elif isinstance(element, chord.Chord):
                    all_notes.append('.'.join(str(n) for n in element.normalOrder))
                    file_notes_count += 1
            
            parsed_files.append({
                "Filename": os.path.basename(file),
                "Notes Count": file_notes_count,
                "Status": "Success ✅"
            })
        except Exception as e:
            parsed_files.append({
                "Filename": os.path.basename(file),
                "Notes Count": 0,
                "Status": f"Failed ❌ ({str(e)})"
            })

    return all_notes, parsed_files


def prepare_sequences(notes, sequence_length=100):
    """Convert notes into input-output pairs for training."""
    unique_notes = sorted(set(notes))
    n_vocab = len(unique_notes)

    # create mapping: note -> number and back
    note_to_int = {note: i for i, note in enumerate(unique_notes)}
    int_to_note = {i: note for i, note in enumerate(unique_notes)}

    # build input sequences and outputs
    X = []
    y = []

    for i in range(len(notes) - sequence_length):
        seq_in = notes[i: i + sequence_length]
        seq_out = notes[i + sequence_length]
        X.append([note_to_int[n] for n in seq_in])
        y.append(note_to_int[seq_out])

    n_patterns = len(X)
    
    # Reshape and normalize for LSTM
    X_reshaped = np.reshape(X, (n_patterns, sequence_length, 1))
    X_normalized = X_reshaped / float(n_vocab)
    
    # One-hot encode targets
    y_encoded = to_categorical(y, num_classes=n_vocab)

    return X_normalized, y_encoded, note_to_int, int_to_note, n_vocab


def build_lstm_model(sequence_length, n_vocab):
    """Build the LSTM model using Keras Sequential API."""
    model = Sequential([
        # First LSTM layer
        LSTM(256, input_shape=(sequence_length, 1), return_sequences=True),
        BatchNormalization(),
        Dropout(0.3),

        # Second LSTM layer
        LSTM(512, return_sequences=True),
        BatchNormalization(),
        Dropout(0.3),

        # Third LSTM layer
        LSTM(256),
        BatchNormalization(),
        Dropout(0.3),

        # Dense layers
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(n_vocab, activation='softmax')
    ])

    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam'
    )
    return model


class StreamlitTrainingCallback(keras.callbacks.Callback):
    """Custom Keras callback to feed live training metrics to Streamlit UI."""
    def __init__(self, epochs, progress_bar, progress_text, loss_placeholder):
        super().__init__()
        self.epochs = epochs
        self.progress_bar = progress_bar
        self.progress_text = progress_text
        self.loss_placeholder = loss_placeholder
        self.losses = []
        self.epochs_completed = 0

    def on_epoch_end(self, epoch, logs=None):
        self.epochs_completed = epoch + 1
        loss = logs.get('loss')
        self.losses.append(loss)
        
        # Update progress UI elements
        percent = self.epochs_completed / self.epochs
        self.progress_bar.progress(percent)
        self.progress_text.text(f"Training Progress: Epoch {self.epochs_completed}/{self.epochs} | Current Loss: {loss:.4f}")
        
        # Plot and update live loss graph
        df_loss = pd.DataFrame({"Loss": self.losses})
        self.loss_placeholder.line_chart(df_loss)


def generate_music_sequence(model, note_to_int, int_to_note, notes, n_vocab,
                             sequence_length=100, num_notes=200, temperature=1.0):
    """Generate notes from the trained model using a random seed."""
    start = random.randint(0, len(notes) - sequence_length - 1)
    seed_sequence = notes[start: start + sequence_length]

    pattern = [note_to_int[n] for n in seed_sequence]
    generated_notes = []

    progress_placeholder = st.empty()
    gen_progress = progress_placeholder.progress(0.0)

    for i in range(num_notes):
        # Prepare input shape (1, seq_len, 1) and normalize
        input_seq = np.reshape(pattern, (1, len(pattern), 1))
        input_seq = input_seq / float(n_vocab)

        # Predict next note probabilities
        prediction = model.predict(input_seq, verbose=0)

        # Apply temperature scaling for creative diversity
        # Add tiny epsilon to avoid division by zero or log(0)
        prediction = np.log(prediction + 1e-8) / temperature
        exp_preds = np.exp(prediction)
        prediction = exp_preds / np.sum(exp_preds)
        
        # Sample next note based on probabilities
        next_index = np.random.choice(len(prediction[0]), p=prediction[0])
        next_note = int_to_note[next_index]
        generated_notes.append(next_note)

        # Slide sequence window
        pattern.append(next_index)
        pattern = pattern[1:]

        # Update generator progress bar periodically
        if (i + 1) % 10 == 0 or i == num_notes - 1:
            gen_progress.progress((i + 1) / num_notes)

    progress_placeholder.empty()
    return generated_notes


def split_lyrics_to_syllables(lyrics_text):
    """Split text into words and approximate syllables for music alignment."""
    if not lyrics_text or not lyrics_text.strip():
        return []
        
    words = lyrics_text.strip().split()
    syllables = []
    
    # Simple syllable counting/splitting helper
    vowels = "aeiouy"
    for word in words:
        clean = "".join(c for c in word if c.isalnum())
        if not clean:
            continue
            
        # Count vowels to estimate syllable count
        count = 0
        if clean[0].lower() in vowels:
            count += 1
        for index in range(1, len(clean)):
            if clean[index].lower() in vowels and clean[index - 1].lower() not in vowels:
                count += 1
        if clean.endswith("e"):
            count -= 1
        if clean.endswith("le") and len(clean) > 2 and clean[-3].lower() not in vowels:
            count += 1
        if count <= 0:
            count = 1
            
        if count == 1:
            syllables.append(word)
        else:
            # Roughly split word into equal parts based on syllable count
            part_len = max(1, len(word) // count)
            for j in range(count):
                if j == count - 1:
                    syllables.append(word[j*part_len:])
                else:
                    syllables.append(word[j*part_len : (j+1)*part_len] + "-")
                    
    return syllables


def note_name_to_midi(note_name):
    """Convert music21 note name to MIDI number (e.g. 'C4' -> 60, 'F#3' -> 54)."""
    if not note_name:
        return 60
    if '.' in note_name:
        note_name = note_name.split('.')[0]
    if note_name.isdigit():
        return 60 + int(note_name)
    pitch_map = {
        'C': 0, 'C#': 1, 'D-': 1, 'D': 2, 'D#': 3, 'E-': 3, 'E': 4,
        'F': 5, 'F#': 6, 'G-': 6, 'G': 7, 'G#': 8, 'A-': 8, 'A': 9,
        'A#': 10, 'B-': 10, 'B': 11
    }
    letter = ""
    octave = 4
    accidental = ""
    if len(note_name) > 0:
        letter = note_name[0].upper()
    if len(note_name) > 1:
        if note_name[1] in ['#', '-']:
            accidental = note_name[1]
            if len(note_name) > 2:
                try:
                    octave = int(note_name[2:])
                except ValueError:
                    pass
        else:
            try:
                octave = int(note_name[1:])
            except ValueError:
                pass
    base = pitch_map.get(letter + accidental, 0)
    return (octave + 1) * 12 + base


def synthesize_singing_voice(generated_notes, syllables, base_midi_note=60, output_file=None):
    """Synthesize a robotic singing voice from notes and syllables using pyttsx3 and pitch shifting."""
    if not generated_notes or not syllables:
        return None

    if output_file is None:
        output_file = os.path.join(SCRIPT_DIR, "generated_vocals.wav")

    # Initialize COM on Windows for thread-safety
    try:
        import pythoncom
        pythoncom.CoInitialize()
    except Exception:
        pass

    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
    except Exception as e:
        print(f"Failed to initialize pyttsx3: {e}")
        return None

    import scipy.io.wavfile as wavfile
    import scipy.signal as signal

    # Filter out notes that are chords or pitch numbers for cleaner alignment
    flat_notes = []
    for n in generated_notes:
        if '.' in n:
            flat_notes.append(n.split('.')[0])
        else:
            flat_notes.append(n)

    # 1. Synthesize and cache unique syllables
    syl_audio_cache = {}
    unique_syls = set(syllables)
    
    progress_bar = st.progress(0.0)
    status_text = st.empty()
    status_text.text("Synthesizing AI singing vocal syllables...")

    temp_folder = os.path.join(SCRIPT_DIR, "temp_vocals")
    os.makedirs(temp_folder, exist_ok=True)

    for idx, syl in enumerate(unique_syls):
        progress_bar.progress((idx + 1) / len(unique_syls))
        speech_text = syl.replace("-", "").strip()
        if not speech_text:
            continue
            
        temp_wav_path = os.path.join(temp_folder, f"syl_{idx}.wav")
        try:
            engine.save_to_file(speech_text, temp_wav_path)
            engine.runAndWait()
            
            if os.path.exists(temp_wav_path):
                sr, data = wavfile.read(temp_wav_path)
                if len(data.shape) > 1:
                    data = data[:, 0]
                
                data_float = data.astype(np.float32)
                if data.dtype == np.int16:
                    data_float /= 32768.0
                elif data.dtype == np.int32:
                    data_float /= 2147483648.0
                    
                syl_audio_cache[syl] = (sr, data_float)
        except Exception as e:
            print(f"Failed to synthesize syllable '{syl}': {e}")
        finally:
            if os.path.exists(temp_wav_path):
                try:
                    os.remove(temp_wav_path)
                except Exception:
                    pass

    try:
        # Clean up temp folder files and then folder
        for f in os.listdir(temp_folder):
            os.remove(os.path.join(temp_folder, f))
        os.rmdir(temp_folder)
    except Exception:
        pass

    progress_bar.empty()
    status_text.empty()

    if not syl_audio_cache:
        return None

    sample_rate = next(iter(syl_audio_cache.values()))[0]

    # 2. Build the master audio track
    num_singing_notes = min(len(flat_notes), len(syllables))
    total_duration = num_singing_notes * 0.5 + 1.5
    total_samples = int(total_duration * sample_rate)
    master_audio = np.zeros(total_samples, dtype=np.float32)

    for idx in range(num_singing_notes):
        note_name = flat_notes[idx]
        syl = syllables[idx]

        if syl not in syl_audio_cache:
            continue

        sr, syl_data = syl_audio_cache[syl]
        note_midi = note_name_to_midi(note_name)
        semitones = note_midi - base_midi_note

        # Pitch shift syllable using resampling
        if semitones == 0:
            shifted = syl_data
        else:
            factor = 2 ** (semitones / 12.0)
            num_samples = int(len(syl_data) / factor)
            if num_samples > 0:
                shifted = signal.resample(syl_data, num_samples)
            else:
                shifted = syl_data

        # Determine start sample (0.5s per note offset)
        start_sample = int(idx * 0.5 * sample_rate)
        end_sample = start_sample + len(shifted)

        # Mix (sum) into the master audio buffer
        if end_sample <= len(master_audio):
            master_audio[start_sample:end_sample] += shifted
        else:
            avail_len = len(master_audio) - start_sample
            if avail_len > 0:
                master_audio[start_sample:] += shifted[:avail_len]

    # Normalize final audio to avoid clipping
    max_val = np.max(np.abs(master_audio))
    if max_val > 0:
        master_audio = master_audio / max_val * 0.9

    # Convert to int16 WAV and save
    master_audio_int16 = (master_audio * 32767.0).astype(np.int16)
    wavfile.write(output_file, sample_rate, master_audio_int16)
    return output_file


def save_notes_to_midi(generated_notes, output_file=None, syllables=None):
    """Convert generated list of notes and chords back into a MIDI file with lyrics."""
    if output_file is None:
        output_file = os.path.join(SCRIPT_DIR, "generated_music.mid")
    output_notes = []
    offset = 0.0
    syl_idx = 0

    for pattern in generated_notes:
        # Check if element represents a chord
        if '.' in pattern or pattern.isdigit():
            chord_notes = pattern.split('.')
            notes_in_chord = []
            for n in chord_notes:
                try:
                    new_note = note.Note(int(n))
                    new_note.storedInstrument = instrument.Piano()
                    notes_in_chord.append(new_note)
                except Exception:
                    pass
            if notes_in_chord:
                # Add lyric to first note of the chord if available
                if syllables and syl_idx < len(syllables):
                    notes_in_chord[0].lyric = syllables[syl_idx]
                    syl_idx += 1
                new_chord = chord.Chord(notes_in_chord)
                new_chord.offset = offset
                output_notes.append(new_chord)
        else:
            # Single note
            try:
                new_note = note.Note(pattern)
                new_note.offset = offset
                new_note.storedInstrument = instrument.Piano()
                
                # Add lyric if available
                if syllables and syl_idx < len(syllables):
                    new_note.lyric = syllables[syl_idx]
                    syl_idx += 1
                    
                output_notes.append(new_note)
            except Exception:
                pass

        # Shift offset time forward
        offset += 0.5

    # Create music21 Stream and write to file
    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp=output_file)


def get_midi_player_html(midi_file_path):
    """Read a MIDI file, encode in Base64, and embed the html-midi-player."""
    with open(midi_file_path, "rb") as f:
        midi_bytes = f.read()
    
    b64_midi = base64.b64encode(midi_bytes).decode('utf-8')
    midi_data_url = f"data:audio/midi;base64,{b64_midi}"
    
    html_code = f"""
    <div style="background-color: #1e1e2f; padding: 20px; border-radius: 12px; font-family: sans-serif; text-align: center; color: white;">
        <h4 style="margin-top: 0; color: #a18cd1; font-weight: 500;">🎵 Interactive MIDI Synthesizer & Piano Roll</h4>
        
        <!-- Web Components from CDN -->
        <script src="https://cdn.jsdelivr.net/combine/npm/tone@14.7.58,npm/@magenta/music@1.23.1/es6/core.js,npm/html-midi-player@1.5.0"></script>
        
        <!-- Player -->
        <midi-player
            src="{midi_data_url}"
            sound-font="https://storage.googleapis.com/magentadata/js/soundfonts/sgm_plus"
            visualizer="#myVisualizer"
            style="width: 100%; max-width: 600px; margin: 15px auto; display: block; border-radius: 8px; background: #2b2b3d; padding: 10px; box-shadow: inset 0 2px 5px rgba(0,0,0,0.5);">
        </midi-player>
        
        <!-- Visualizer Container -->
        <div style="background-color: #121218; border-radius: 8px; border: 1px solid #3a3a55; padding: 5px; margin-top: 15px; overflow-x: auto;">
            <midi-visualizer
                type="piano-roll"
                id="myVisualizer"
                style="width: 100%; min-width: 500px; height: 180px;">
            </midi-visualizer>
        </div>
        
        <p style="font-size: 11px; color: #8c8ca3; margin-bottom: 0; margin-top: 10px;">
            Synthesized live with Soundfont (SGM Plus) and Tone.js. Scroll and drag inside the piano roll to explore notes!
        </p>
    </div>
    """
    return html_code


# ============================================================
# APP LAYOUT & NAVIGATION
# ============================================================

# Header Banner
st.markdown("""
<div class="main-header">
    <h1>🎵 AI Melody Generator</h1>
    <p>CodeAlpha AI Internship - Task 3 | Deep Learning Music Composition using LSTMs</p>
</div>
""", unsafe_allow_html=True)

# Tabs
tab_explore, tab_train, tab_generate = st.tabs([
    "📂 1. Explore & Analyze MIDI Data", 
    "⚙️ 2. Build & Train LSTM Model", 
    "🎹 3. Generate & Play AI Music"
])

# Read active list of files
midi_files_list = glob.glob(os.path.join(MIDI_FOLDER, "*.mid"))

# ============================================================
# TAB 1: EXPLORE & ANALYZE MIDI DATA
# ============================================================
with tab_explore:
    st.header("Explore and Analyze MIDI Files")
    
    col_upload, col_list = st.columns([1, 1])
    
    with col_upload:
        st.subheader("Upload Custom MIDI Files")
        uploaded_files = st.file_uploader(
            "Upload `.mid` files to add to the training corpus:", 
            type=["mid", "midi"], 
            accept_multiple_files=True
        )
        if uploaded_files:
            for uploaded_file in uploaded_files:
                file_path = os.path.join(MIDI_FOLDER, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
            st.success(f"Saved {len(uploaded_files)} file(s) into '{MIDI_FOLDER}'! Click refresh below to rebuild cache.")
            st.button("🔄 Refresh Data Cache", on_click=load_and_cache_notes.clear)

    with col_list:
        st.subheader("Manage Current MIDI Library")
        st.write(f"Folder `{MIDI_FOLDER}/` has **{len(midi_files_list)}** MIDI file(s).")
        
        if st.button("🗑️ Clear Custom MIDI Files (Keep bach_*.mid)"):
            deleted_count = 0
            for f in glob.glob(os.path.join(MIDI_FOLDER, "*")):
                if os.path.basename(f).startswith("bach_") and os.path.basename(f).endswith(".mid"):
                    continue
                try:
                    os.remove(f)
                    deleted_count += 1
                except Exception:
                    pass
            st.success(f"Removed {deleted_count} file(s).")
            st.button("🔄 Refresh Data Cache", key="refresh_clear", on_click=load_and_cache_notes.clear)

    st.divider()

    # Load and process files
    with st.spinner("Analyzing MIDI notes and chords..."):
        notes, parsed_summary = load_and_cache_notes(MIDI_FOLDER)

    if notes:
        st.subheader("Parsed MIDI Summary")
        
        # Display Stats Cards
        card_col1, card_col2, card_col3 = st.columns(3)
        with card_col1:
            st.markdown(f"""
            <div class="stats-card">
                <div class="stats-number">{len(parsed_summary)}</div>
                <div class="stats-label">Files Checked</div>
            </div>
            """, unsafe_allow_html=True)
        with card_col2:
            st.markdown(f"""
            <div class="stats-card">
                <div class="stats-number">{len(notes)}</div>
                <div class="stats-label">Total Notes/Chords parsed</div>
            </div>
            """, unsafe_allow_html=True)
        with card_col3:
            st.markdown(f"""
            <div class="stats-card">
                <div class="stats-number">{len(set(notes))}</div>
                <div class="stats-label">Vocabulary Size (Unique notes)</div>
            </div>
            """, unsafe_allow_html=True)

        st.dataframe(pd.DataFrame(parsed_summary), use_container_width=True)

        # Plot note frequency distribution
        st.subheader("Note and Chord Frequency Analysis")
        note_series = pd.Series(notes)
        note_counts = note_series.value_counts().head(20)

        fig, ax = plt.subplots(figsize=(10, 4))
        # Style matplotlib plot
        ax.bar(note_counts.index, note_counts.values, color='#667eea', edgecolor='#4a5568')
        ax.set_title("Top 20 Most Frequent Notes & Chords", color='#4a5568', fontsize=12, fontweight='bold')
        ax.set_ylabel("Count", color='#4a5568')
        ax.set_xlabel("Note / Chord ID (music21 representation)", color='#4a5568')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)

    else:
        st.warning("No MIDI files have been parsed. Make sure there are `.mid` files in the `midi_songs/` folder.")


# ============================================================
# TAB 2: BUILD & TRAIN LSTM MODEL
# ============================================================
with tab_train:
    st.header("Configure and Train Keras LSTM Network")
    
    if not notes:
        st.error("Please load some MIDI files in Tab 1 before training.")
    else:
        # Configurations
        col_cfg, col_summary = st.columns([1, 1])

        with col_cfg:
            st.subheader("Hyperparameters")
            seq_len = st.slider("Sequence Length (Notes seed size)", min_value=10, max_value=150, value=100, step=5)
            epochs = st.slider("Training Epochs", min_value=1, max_value=100, value=20, step=1)
            batch_size = st.selectbox("Batch Size", options=[32, 64, 128, 256], index=1)
            
            st.info("💡 LSTMs model sequences. A sequence length of 100 means the model looks at 100 preceding notes to predict the 101st note.")
        
        with col_summary:
            st.subheader("Data Suitability")
            total_notes = len(notes)
            if total_notes <= seq_len:
                st.error(f"Not enough notes ({total_notes}) to build sequences with length {seq_len}. Add more MIDI files or reduce Sequence Length.")
                can_train = False
            else:
                st.success(f"Ready to build training sequences! Total samples: {total_notes - seq_len}")
                can_train = True

        if can_train:
            st.divider()
            
            # Show architecture
            with st.expander("🔍 View LSTM Network Architecture Details"):
                st.code("""
Sequential([
    # Layer 1: LSTM (256 units, return sequences) + Batch Normalization + Dropout (0.3)
    LSTM(256, input_shape=(sequence_length, 1), return_sequences=True),
    
    # Layer 2: LSTM (512 units, return sequences) + Batch Normalization + Dropout (0.3)
    LSTM(512, return_sequences=True),
    
    # Layer 3: LSTM (256 units) + Batch Normalization + Dropout (0.3)
    LSTM(256),
    
    # Layer 4: Fully Connected Dense Layer (256 units, ReLU) + Dropout (0.3)
    Dense(256, activation='relu'),
    
    # Output Layer: Dense Layer (n_vocab units, Softmax) -> probabilities for each note
    Dense(n_vocab, activation='softmax')
])
                """)
            
            col_actions, col_logs = st.columns([1, 2])
            
            with col_actions:
                st.subheader("Training Actions")
                train_clicked = st.button("🚀 Start Training Model", type="primary")
                
                weights_path = os.path.join(SCRIPT_DIR, "music_model_weights.weights.h5")
                notes_pkl_path = os.path.join(SCRIPT_DIR, "notes.pkl")
                
                if os.path.exists(weights_path):
                    st.success("Existing model weights `music_model_weights.weights.h5` found! You can generate melody directly in Tab 3 or retrain.")
                else:
                    st.info("No saved model weights found. Please train first.")

            with col_logs:
                if train_clicked:
                    st.subheader("Training Progress")
                    progress_bar = st.progress(0.0)
                    progress_text = st.empty()
                    progress_text.text("Preparing sequences and mapping vocabulary...")
                    
                    # Prepare sequences
                    X, y, note_to_int, int_to_note, n_vocab = prepare_sequences(notes, seq_len)
                    
                    progress_text.text("Compiling LSTM model...")
                    model = build_lstm_model(seq_len, n_vocab)
                    
                    progress_text.text("Starting training loop... coffee break time! ☕")
                    loss_chart = st.empty()
                    
                    streamlit_callback = StreamlitTrainingCallback(epochs, progress_bar, progress_text, loss_chart)
                    
                    # Model Checkpoint
                    checkpoint = keras.callbacks.ModelCheckpoint(
                        weights_path,
                        monitor='loss',
                        save_best_only=True,
                        save_weights_only=True,
                        mode='min',
                        verbose=0
                    )
                    
                    # Save vocabulary mapping
                    with open(notes_pkl_path, "wb") as f:
                        pickle.dump(notes, f)
                    
                    # Start fit
                    start_time = time.time()
                    model.fit(
                        X, y,
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=[checkpoint, streamlit_callback],
                        verbose=0
                    )
                    end_time = time.time()
                    
                    st.success(f"🎉 Training completed in {end_time - start_time:.1f} seconds! Saved weights to `{weights_path}` and notes map to `{notes_pkl_path}`.")
                    st.button("🔄 Refresh Application State")


# ============================================================
# TAB 3: GENERATE & PLAY AI MUSIC
# ============================================================
with tab_generate:
    st.header("Generate and Play AI Music Melodies")
    
    weights_path = os.path.join(SCRIPT_DIR, "music_model_weights.weights.h5")
    notes_pkl_path = os.path.join(SCRIPT_DIR, "notes.pkl")
    
    if not os.path.exists(weights_path) or not os.path.exists(notes_pkl_path):
        st.warning("⚠️ No trained model or vocabulary map found! Please complete model training in Tab 2 first.")
    else:
        # Load vocabulary mappings
        with open(notes_pkl_path, "rb") as f:
            saved_notes = pickle.load(f)
            
        unique_notes = sorted(set(saved_notes))
        n_vocab = len(unique_notes)
        note_to_int = {note: i for i, note in enumerate(unique_notes)}
        int_to_note = {i: note for i, note in enumerate(unique_notes)}
        
        col_gen_cfg, col_gen_play = st.columns([1, 2])
        
        with col_gen_cfg:
            st.subheader("Generation Settings")
            gen_notes_count = st.slider("Melody Length (Number of notes to generate)", min_value=50, max_value=500, value=200, step=10)
            gen_seq_len = st.slider("Seed Sequence Length", min_value=10, max_value=150, value=100, step=5)
            
            # Temperature explanation
            st.markdown("""
            ##### Temperature (Creativity Index)
            * **Lower values (e.g. 0.5 - 0.8)**: More repetitive, conservative, and predictable music.
            * **1.0**: Normal neural network output.
            * **Higher values (e.g. 1.2 - 1.8)**: More experimental, random, and creative music.
            """)
            temp = st.slider("Temperature", min_value=0.2, max_value=2.0, value=1.0, step=0.1)
            
            lyrics_input = st.text_area(
                "📝 Sync Lyrics with Melody (Optional)", 
                placeholder="Type or paste your lyrics here...\nSyllables will be synced note-by-note into the MIDI file!",
                help="Each syllable of the lyrics will be assigned to a sequential note of the generated music melody."
            )
            
            vocal_base_key = st.selectbox(
                "🎤 AI Singer Base Pitch Key",
                options=["C3 (Low)", "E3", "G3", "C4 (Mid)", "E4", "G4 (High)"],
                index=3,
                help="Change this to shift the pitch of the AI singing voice up or down."
            )
            
            st.divider()
            generate_clicked = st.button("🎹 Generate AI Melody", type="primary", use_container_width=True)

        with col_gen_play:
            if generate_clicked:
                st.subheader("Generating Music Sequence...")
                
                with st.spinner("Loading weights and initializing LSTM network..."):
                    # Build and load weights
                    model = build_lstm_model(gen_seq_len, n_vocab)
                    model.load_weights(weights_path)
                
                with st.spinner("Generating note-by-note compositions using deep learning..."):
                    generated_notes = generate_music_sequence(
                        model, note_to_int, int_to_note, saved_notes, n_vocab,
                        sequence_length=gen_seq_len, num_notes=gen_notes_count, temperature=temp
                    )
                
                st.success(f"Generated {len(generated_notes)} notes and chords successfully!")
                
                # Parse vocal base MIDI key
                base_midi_map = {
                    "C3 (Low)": 48,
                    "E3": 52,
                    "G3": 55,
                    "C4 (Mid)": 60,
                    "E4": 64,
                    "G4 (High)": 67
                }
                vocal_base_midi = base_midi_map.get(vocal_base_key, 60)

                # Save to MIDI file
                output_midi_file = os.path.join(SCRIPT_DIR, "generated_music.mid")
                with st.spinner("Converting notes back into MIDI structure..."):
                    syllables = split_lyrics_to_syllables(lyrics_input) if lyrics_input else None
                    save_notes_to_midi(generated_notes, output_midi_file, syllables=syllables)
                
                # Synthesize vocals if lyrics are provided
                vocal_audio_file = os.path.join(SCRIPT_DIR, "generated_vocals.wav")
                vocals_synthesized = False
                if lyrics_input and syllables:
                    with st.spinner("Synthesizing AI singing voice audio..."):
                        res = synthesize_singing_voice(
                            generated_notes, 
                            syllables, 
                            base_midi_note=vocal_base_midi, 
                            output_file=vocal_audio_file
                        )
                        if res:
                            vocals_synthesized = True

                # Show download button
                with open(output_midi_file, "rb") as f:
                    midi_data = f.read()
                
                st.download_button(
                    label="📥 Download Generated MIDI File (with lyrics)",
                    data=midi_data,
                    file_name="ai_music_generator_melody.mid",
                    mime="audio/midi",
                    use_container_width=True
                )
                
                st.divider()
                
                # Render interactive player
                with st.spinner("Loading interactive synthesizers..."):
                    midi_player_html = get_midi_player_html(output_midi_file)
                    st.components.v1.html(midi_player_html, height=400, scrolling=False)
                
                # Render Vocal Player if voice was synthesized
                if vocals_synthesized:
                    st.subheader("🎤 AI Vocal Synthesis Preview")
                    st.write("Listen to the synthesized robotic voice singing your lyrics at the correct pitch:")
                    
                    with open(vocal_audio_file, "rb") as f:
                        vocal_data = f.read()
                    
                    # Columns for play and download
                    play_col, dl_col = st.columns([2, 1])
                    with play_col:
                        st.audio(vocal_data, format="audio/wav")
                    with dl_col:
                        st.download_button(
                            label="📥 Download Vocals Audio (.wav)",
                            data=vocal_data,
                            file_name="ai_generated_vocals.wav",
                            mime="audio/wav",
                            use_container_width=True
                        )
                    st.divider()

                # Render Lyric-to-Note alignment grid
                if lyrics_input and syllables:
                    st.subheader("🎤 Syllable Alignment Sheet")
                    st.info("Here is how your lyrics have been synced to the generated notes sequence. This metadata is embedded directly inside the downloaded MIDI file!")
                    
                    # Align notes (ignoring chords for clean lyrics display)
                    notes_only = [n for n in generated_notes if not ('.' in n or n.isdigit())]
                    
                    lyric_blocks = []
                    for idx in range(min(len(notes_only), len(syllables))):
                        lyric_blocks.append(f"**{notes_only[idx]}**<br><span style='color:#a18cd1; font-weight:bold; font-size:1.1rem;'>{syllables[idx]}</span>")
                    
                    # Render grid layout in rows of 8 cards
                    cols_per_row = 8
                    for idx in range(0, len(lyric_blocks), cols_per_row):
                        row_items = lyric_blocks[idx : idx + cols_per_row]
                        cols = st.columns(cols_per_row)
                        for col_idx, col in enumerate(cols):
                            if col_idx < len(row_items):
                                col.markdown(f"<div style='text-align:center; background-color:#2b2b3d; padding:10px; border-radius:8px; border:1px solid #4a5568; margin-bottom:10px;'>{row_items[col_idx]}</div>", unsafe_allow_html=True)
                    
            else:
                # Default state if not generated yet
                st.info("Adjust the configurations on the left and click **Generate AI Melody** to start composed melodies!")
                prev_midi_path = os.path.join(SCRIPT_DIR, "generated_music.mid")
                prev_vocal_path = os.path.join(SCRIPT_DIR, "generated_vocals.wav")
                
                if os.path.exists(prev_midi_path):
                    st.subheader("Previous Generated Compositions")
                    st.write("You can listen to or download your previously generated melody:")
                    
                    with open(prev_midi_path, "rb") as f:
                        midi_data = f.read()
                    
                    st.download_button(
                        label="📥 Download Previous MIDI",
                        data=midi_data,
                        file_name="ai_music_generator_melody.mid",
                        mime="audio/midi",
                        use_container_width=True
                    )
                    
                    midi_player_html = get_midi_player_html(prev_midi_path)
                    st.components.v1.html(midi_player_html, height=400, scrolling=False)

                if os.path.exists(prev_vocal_path):
                    st.subheader("🎤 Previous Synthesized Vocals")
                    st.write("Listen to or download your previously generated singing vocal track:")
                    
                    with open(prev_vocal_path, "rb") as f:
                        vocal_data = f.read()
                    
                    play_col, dl_col = st.columns([2, 1])
                    with play_col:
                        st.audio(vocal_data, format="audio/wav")
                    with dl_col:
                        st.download_button(
                            label="📥 Download Previous Vocals (.wav)",
                            data=vocal_data,
                            file_name="ai_generated_vocals.wav",
                            mime="audio/wav",
                            use_container_width=True
                        )
