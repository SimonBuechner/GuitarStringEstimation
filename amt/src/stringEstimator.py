#%%
#%cd /content/amt/src
import os
import sys

from scipy.stats import find_repeats

sys.path.append(os.path.abspath(''))
#dir_path = os.path.dirname(os.path.realpath(__file__))
#os.chdir('./src')
#dir_path_post = os.path.dirname(os.path.realpath(__file__))

from collections import Counter
import argparse
import torch
import torchaudio
import numpy as np

# signal processing imports
import scipy
from scipy import signal
from scipy.io import wavfile
import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use('TkAgg')

from scipy.fft import rfft
from scipy.fft import rfftfreq
from scipy.signal import find_peaks
import scipy as sc


from typing import Tuple, Dict, Literal

from model.init_train import initialize_trainer, update_config
from utils.task_manager import TaskManager
from config.vocabulary import drum_vocab_presets
from utils.utils import str2bool
from utils.utils import Timer
from utils.audio import slice_padded_array
from utils.note2event import mix_notes
from utils.event2note import merge_zipped_note_events_and_ties_to_notes
from utils.utils import write_model_output_as_midi, write_err_cnt_as_json
from model.ymt3 import YourMT3

# Clean Project
plt.close('all')

#%% @title model helper
def load_model_checkpoint(args=None):
    parser = argparse.ArgumentParser(description="YourMT3")
    # General
    parser.add_argument('exp_id', type=str, help='A unique identifier for the experiment is used to resume training. The "@" symbol can be used to load a specific checkpoint.')
    parser.add_argument('-p', '--project', type=str, default='ymt3', help='project name')
    parser.add_argument('-ac', '--audio-codec', type=str, default=None, help='audio codec (default=None). {"spec", "melspec"}. If None, default value defined in config.py will be used.')
    parser.add_argument('-hop', '--hop-length', type=int, default=None, help='hop length in frames (default=None). {128, 300} 128 for MT3, 300 for PerceiverTFIf None, default value defined in config.py will be used.')
    parser.add_argument('-nmel', '--n-mels', type=int, default=None, help='number of mel bins (default=None). If None, default value defined in config.py will be used.')
    parser.add_argument('-if', '--input-frames', type=int, default=None, help='number of audio frames for input segment (default=None). If None, default value defined in config.py will be used.')
    # Model configurations
    parser.add_argument('-sqr', '--sca-use-query-residual', type=str2bool, default=None, help='sca use query residual flag. Default follows config.py')
    parser.add_argument('-enc', '--encoder-type', type=str, default=None, help="Encoder type. 't5' or 'perceiver-tf' or 'conformer'. Default is 't5', following config.py.")
    parser.add_argument('-dec', '--decoder-type', type=str, default=None, help="Decoder type. 't5' or 'multi-t5'. Default is 't5', following config.py.")
    parser.add_argument('-preenc', '--pre-encoder-type', type=str, default='default', help="Pre-encoder type. None or 'conv' or 'default'. By default, t5_enc:None, perceiver_tf_enc:conv, conformer:None")
    parser.add_argument('-predec', '--pre-decoder-type', type=str, default='default', help="Pre-decoder type. {None, 'linear', 'conv1', 'mlp', 'group_linear'} or 'default'. Default is {'t5': None, 'perceiver-tf': 'linear', 'conformer': None}.")
    parser.add_argument('-cout', '--conv-out-channels', type=int, default=None, help='Number of filters for pre-encoder conv layer. Default follows "model_cfg" of config.py.')
    parser.add_argument('-tenc', '--task-cond-encoder', type=str2bool, default=True, help='task conditional encoder (default=True). True or False')
    parser.add_argument('-tdec', '--task-cond-decoder', type=str2bool, default=True, help='task conditional decoder (default=True). True or False')
    parser.add_argument('-df', '--d-feat', type=int, default=None, help='Audio feature will be projected to this dimension for Q,K,V of T5 or K,V of Perceiver (default=None). If None, default value defined in config.py will be used.')
    parser.add_argument('-pt', '--pretrained', type=str2bool, default=False, help='pretrained T5(default=False). True or False')
    parser.add_argument('-b', '--base-name', type=str, default="google/t5-v1_1-small", help='base model name (default="google/t5-v1_1-small")')
    parser.add_argument('-epe', '--encoder-position-encoding-type', type=str, default='default', help="Positional encoding type of encoder. By default, pre-defined PE for T5 or Perceiver-TF encoder in config.py. For T5: {'sinusoidal', 'trainable'}, conformer: {'rotary', 'trainable'}, Perceiver-TF: {'trainable', 'rope', 'alibi', 'alibit', 'None', '0', 'none', 'tkd', 'td', 'tk', 'kdt'}.")
    parser.add_argument('-dpe', '--decoder-position-encoding-type', type=str, default='default', help="Positional encoding type of decoder. By default, pre-defined PE for T5 in config.py. {'sinusoidal', 'trainable'}.")
    parser.add_argument('-twe', '--tie-word-embedding', type=str2bool, default=None, help='tie word embedding (default=None). If None, default value defined in config.py will be used.')
    parser.add_argument('-el', '--event-length', type=int, default=None, help='event length (default=None). If None, default value defined in model cfg of config.py will be used.')
    # Perceiver-TF configurations
    parser.add_argument('-dl', '--d-latent', type=int, default=None, help='Latent dimension of Perceiver. On T5, this will be ignored (default=None). If None, default value defined in config.py will be used.')
    parser.add_argument('-nl', '--num-latents', type=int, default=None, help='Number of latents of Perceiver. On T5, this will be ignored (default=None). If None, default value defined in config.py will be used.')
    parser.add_argument('-dpm', '--perceiver-tf-d-model', type=int, default=None, help='Perceiver-TF d_model (default=None). If None, default value defined in config.py will be used.')
    parser.add_argument('-npb', '--num-perceiver-tf-blocks', type=int, default=None, help='Number of blocks of Perceiver-TF. On T5, this will be ignored (default=None). If None, default value defined in config.py.')
    parser.add_argument('-npl', '--num-perceiver-tf-local-transformers-per-block', type=int, default=None, help='Number of local layers per block of Perceiver-TF. On T5, this will be ignored (default=None). If None, default value defined in config.py will be used.')
    parser.add_argument('-npt', '--num-perceiver-tf-temporal-transformers-per-block', type=int, default=None, help='Number of temporal layers per block of Perceiver-TF. On T5, this will be ignored (default=None). If None, default value defined in config.py will be used.')
    parser.add_argument('-atc', '--attention-to-channel', type=str2bool, default=None, help='Attention to channel flag of Perceiver-TF. On T5, this will be ignored (default=None). If None, default value defined in config.py will be used.')
    parser.add_argument('-ln', '--layer-norm-type', type=str, default=None, help='Layer normalization type (default=None). {"layer_norm", "rms_norm"}. If None, default value defined in config.py will be used.')
    parser.add_argument('-ff', '--ff-layer-type', type=str, default=None, help='Feed forward layer type (default=None). {"mlp", "moe", "gmlp"}. If None, default value defined in config.py will be used.')
    parser.add_argument('-wf', '--ff-widening-factor', type=int, default=None, help='Feed forward layer widening factor for MLP/MoE/gMLP (default=None). If None, default value defined in config.py will be used.')
    parser.add_argument('-nmoe', '--moe-num-experts', type=int, default=None, help='Number of experts for MoE (default=None). If None, default value defined in config.py will be used.')
    parser.add_argument('-kmoe', '--moe-topk', type=int, default=None, help='Top-k for MoE (default=None). If None, default value defined in config.py will be used.')
    parser.add_argument('-act', '--hidden-act', type=str, default=None, help='Hidden activation function (default=None). {"gelu", "silu", "relu", "tanh"}. If None, default value defined in config.py will be used.')
    parser.add_argument('-rt', '--rotary-type', type=str, default=None, help='Rotary embedding type expressed in three letters. e.g. ppl: "pixel" for SCA and latents, "lang" for temporal transformer. If None, use config.')
    parser.add_argument('-rk', '--rope-apply-to-keys', type=str2bool, default=None, help='Apply rope to keys (default=None). If None, use config.')
    parser.add_argument('-rp', '--rope-partial-pe', type=str2bool, default=None, help='Whether to apply RoPE to partial positions (default=None). If None, use config.')
    # Decoder configurations
    parser.add_argument('-dff', '--decoder-ff-layer-type', type=str, default=None, help='Feed forward layer type of decoder (default=None). {"mlp", "moe", "gmlp"}. If None, default value defined in config.py will be used.')
    parser.add_argument('-dwf', '--decoder-ff-widening-factor', type=int, default=None, help='Feed forward layer widening factor for decoder MLP/MoE/gMLP (default=None). If None, default value defined in config.py will be used.')
    # Task and Evaluation configurations
    parser.add_argument('-tk', '--task', type=str, default='mt3_full_plus', help='tokenizer type (default=mt3_full_plus). See config/task.py for more options.')
    parser.add_argument('-epv', '--eval-program-vocab', type=str, default=None, help='evaluation vocabulary (default=None). If None, default vocabulary of the data preset will be used.')
    parser.add_argument('-edv', '--eval-drum-vocab', type=str, default=None, help='evaluation vocabulary for drum (default=None). If None, default vocabulary of the data preset will be used.')
    parser.add_argument('-etk', '--eval-subtask-key', type=str, default='default', help='evaluation subtask key (default=default). See config/task.py for more options.')
    parser.add_argument('-t', '--onset-tolerance', type=float, default=0.05, help='onset tolerance (default=0.05).')
    parser.add_argument('-os', '--test-octave-shift', type=str2bool, default=False, help='test optimal octave shift (default=False). True or False')
    parser.add_argument('-w', '--write-model-output', type=str2bool, default=True, help='write model test output to file (default=False). True or False')
    # Trainer configurations
    parser.add_argument('-pr','--precision', type=str, default="bf16-mixed", help='precision (default="bf16-mixed") {32, 16, bf16, bf16-mixed}')
    parser.add_argument('-st', '--strategy', type=str, default='auto', help='strategy (default=auto). auto or deepspeed or ddp')
    parser.add_argument('-n', '--num-nodes', type=int, default=1, help='number of nodes (default=1)')
    parser.add_argument('-g', '--num-gpus', type=str, default='auto', help='number of gpus (default="auto")')
    parser.add_argument('-wb', '--wandb-mode', type=str, default="disabled", help='wandb mode for logging (default=None). "disabled" or "online" or "offline". If None, default value defined in config.py will be used.')
    # Debug
    parser.add_argument('-debug', '--debug-mode', type=str2bool, default=False, help='debug mode (default=False). True or False')
    parser.add_argument('-tps', '--test-pitch-shift', type=int, default=None, help='use pitch shift when testing. debug-purpose only. (default=None). semitone in int.')
    args = parser.parse_args(args)
    # yapf: enable
    if torch.__version__ >= "1.13":
        torch.set_float32_matmul_precision("high")
    args.epochs = None

    # Initialize and update config
    _, _, dir_info, shared_cfg = initialize_trainer(args, stage='test')
    shared_cfg, audio_cfg, model_cfg = update_config(args, shared_cfg, stage='test')

    if args.eval_drum_vocab != None:  # override eval_drum_vocab
        eval_drum_vocab = drum_vocab_presets[args.eval_drum_vocab]

    # Initialize task manager
    tm = TaskManager(task_name=args.task,
                     max_shift_steps=int(shared_cfg["TOKENIZER"]["max_shift_steps"]),
                     debug_mode=args.debug_mode)
    print(f"Task: {tm.task_name}, Max Shift Steps: {tm.max_shift_steps}")

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model
    model = YourMT3(
        audio_cfg=audio_cfg,
        model_cfg=model_cfg,
        shared_cfg=shared_cfg,
        optimizer=None,
        task_manager=tm,  # tokenizer is a member of task_manager
        eval_subtask_key=args.eval_subtask_key,
        write_output_dir=dir_info["lightning_dir"] if args.write_model_output or args.test_octave_shift else None
        ).to(device)
    checkpoint = torch.load(dir_info["last_ckpt_path"], map_location=device)
    state_dict = checkpoint['state_dict']
    new_state_dict = {k: v for k, v in state_dict.items() if 'pitchshift' not in k}
    
    model.load_state_dict(new_state_dict, strict=False)

    return model.eval()


def transcribe(model, audio_info):
    t = Timer()

    # Converting Audio
    t.start()
    audio, sr = torchaudio.load(uri=audio_info['filepath'])
    audio = torch.mean(audio, dim=0).unsqueeze(0)
    audio = torchaudio.functional.resample(audio, sr, model.audio_cfg['sample_rate'])
    audio_segments = slice_padded_array(audio, model.audio_cfg['input_frames'], model.audio_cfg['input_frames'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    audio_segments = torch.from_numpy(audio_segments.astype('float32')).to(device).unsqueeze(1) # (n_seg, 1, seg_sz)
    t.stop(); t.print_elapsed_time("converting audio");

    # Inference
    t.start()
    pred_token_arr, _ = model.inference_file(bsz=8, audio_segments=audio_segments)
    t.stop(); t.print_elapsed_time("model inference");

    # Post-processing
    t.start()
    num_channels = model.task_manager.num_decoding_channels
    n_items = audio_segments.shape[0]
    start_secs_file = [model.audio_cfg['input_frames'] * i / model.audio_cfg['sample_rate'] for i in range(n_items)]
    pred_notes_in_file = []
    n_err_cnt = Counter()
    for ch in range(num_channels):
        pred_token_arr_ch = [arr[:, ch, :] for arr in pred_token_arr]  # (B, L)
        zipped_note_events_and_tie, list_events, ne_err_cnt = model.task_manager.detokenize_list_batches(
            pred_token_arr_ch, start_secs_file, return_events=True)
        pred_notes_ch, n_err_cnt_ch = merge_zipped_note_events_and_ties_to_notes(zipped_note_events_and_tie)
        pred_notes_in_file.append(pred_notes_ch)
        n_err_cnt += n_err_cnt_ch
    pred_notes = mix_notes(pred_notes_in_file)  # This is the mixed notes from all channels


    #######################################################################
    # export tokens (pred_notes) here for further Examination and Evaluation

    # pred_notes ist eine Liste aus Note-Objects. Diese kann einfach auf die nötigen Daten reduziert werden
    # Dann an  process audio zurückgeben



    # Write MIDI
    #output_dir = 'Users/simonbuechner/Documents/Studium/AKT/3.Semester_GRAZ_WS2425/Toningenieur-Projekt/dev/YourMT3_evaluation/amt/content/'
    #print(f"Transcribe working directory: {os.getcwd()}") --> src/
    output_directory = '../content/'

    output_file = write_model_output_as_midi(pred_notes, output_directory,
                              audio_info['track_name'], model.midi_output_inverse_vocab)
    t.stop(); t.print_elapsed_time("post processing");
    #output_file =  os.path.join(output_file, audio_info['track_name']  + '.mid')

    #output_directory = os.path.abspath(midifile)
    #print(f"Resolved output directory: {output_directory}")
    #midifile = os.path.join(midifile, audio_info['track_name'] + '.mid')

    output_file = os.path.abspath(output_file)
    #assert os.path.exists(output_directory)
    assert os.path.exists(output_file)

    return output_file


def prepare_media(source_path_or_url: os.PathLike,
                  source_type: Literal['audio_filepath', 'youtube_url'],
                  delete_video: bool = True) -> Dict:
    """prepare media from source path or youtube, and return audio info"""
    # Get audio_file
    if source_type == 'audio_filepath':
        audio_file = source_path_or_url
    else:
        raise ValueError(source_type)

    # Create info
    info = torchaudio.info(audio_file)
    return {
        "filepath": audio_file,
        "track_name": os.path.basename(audio_file).split('.')[0],
        "sample_rate": int(info.sample_rate),
        "bits_per_sample": int(info.bits_per_sample),
        "num_channels": int(info.num_channels),
        "num_frames": int(info.num_frames),
        "duration": int(info.num_frames / info.sample_rate),
        "encoding": str.lower(info.encoding),
        }

def process_audio(model, audio_filepath):
    if audio_filepath is None:
        return None
    audio_info = prepare_media(audio_filepath, source_type='audio_filepath')
    print(audio_info)
    midifile = transcribe(model, audio_info)

    #debug
    print(f"Checking existence for: {midifile}")
    print(f"Absolute path: {os.path.abspath(midifile)}")
    assert os.path.exists(midifile), f"File does not exist: {midifile}"
    return midifile

def process_audio_notes(model, audio_filepath):
    if audio_filepath is None:
        return None
    audio_info = prepare_media(audio_filepath, source_type='audio_filepath')
    print(audio_info)
    pred_notes = transcribe_notes(model, audio_info)

    # return policy
    return pred_notes


def transcribe_notes(model, audio_info):
    t = Timer()

    # Converting Audio
    t.start()
    audio, sr = torchaudio.load(uri=audio_info['filepath'])
    audio = torch.mean(audio, dim=0).unsqueeze(0)
    audio = torchaudio.functional.resample(audio, sr, model.audio_cfg['sample_rate'])
    audio_segments = slice_padded_array(audio, model.audio_cfg['input_frames'], model.audio_cfg['input_frames'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    audio_segments = torch.from_numpy(audio_segments.astype('float32')).to(device).unsqueeze(1) # (n_seg, 1, seg_sz)
    t.stop(); t.print_elapsed_time("converting audio");

    # Inference
    t.start()
    pred_token_arr, _ = model.inference_file(bsz=8, audio_segments=audio_segments)
    t.stop(); t.print_elapsed_time("model inference");

    # Post-processing
    t.start()
    num_channels = model.task_manager.num_decoding_channels
    n_items = audio_segments.shape[0]
    start_secs_file = [model.audio_cfg['input_frames'] * i / model.audio_cfg['sample_rate'] for i in range(n_items)]
    pred_notes_in_file = []
    n_err_cnt = Counter()
    for ch in range(num_channels):
        pred_token_arr_ch = [arr[:, ch, :] for arr in pred_token_arr]  # (B, L)
        zipped_note_events_and_tie, list_events, ne_err_cnt = model.task_manager.detokenize_list_batches(
            pred_token_arr_ch, start_secs_file, return_events=True)
        pred_notes_ch, n_err_cnt_ch = merge_zipped_note_events_and_ties_to_notes(zipped_note_events_and_tie)
        pred_notes_in_file.append(pred_notes_ch)
        n_err_cnt += n_err_cnt_ch
    pred_notes = mix_notes(pred_notes_in_file)  # This is the mixed notes from all channels


    #######################################################################
    # export tokens (pred_notes) here for further Examination and Evaluation

    # pred_notes ist eine Liste aus Note-Objects. Diese kann einfach auf die nötigen Daten reduziert werden
    # Dann an  process audio zurückgeben

    return pred_notes


def extract_GT(audio_filepath):
    # Root-Ordner extrahieren
    audio_dir = os.path.dirname(os.path.dirname(audio_filepath))
    annotations_dir = os.path.join(audio_dir, "annotation")

    # Audio-Dateiname extrahieren und umwandeln
    base_name, _ = os.path.splitext(os.path.basename(audio_filepath))
    filename = base_name.replace("_mic", "")
    annotation_filename = f"{filename}_notes.npy" # Notes-Data
    annotation_filepath = os.path.join(annotations_dir, annotation_filename)

    # Existenzprüfung
    if not os.path.exists(annotation_filepath):
        print(f"Warnung: Annotation-Datei '{annotation_filepath}' existiert nicht.")
        return None

    assert os.path.exists(annotation_filepath)
    # load annotation
    GT_array = np.load(annotation_filepath, allow_pickle=True)
    #print(GT_array)
    #print(type(GT_array))  # Zeigt den Typ des Objekts
    #print(GT_array.shape)  # Zeigt die Dimensionen des Arrays
    #print(GT_array.dtype)  # Zeigt den Datentyp der Elemente

    data = GT_array.item()  # `.item()` gibt das einzelne Objekt im Array zurück
    # Zugriff auf 'notes'
    notes = data['notes']
    return notes







# %% @title Load Checkpoint
def main():
    # Order

    model_name = 'YPTF.MoE+Multi (noPS)' # @param ["YMT3+", "YPTF+Single (noPS)", "YPTF+Multi (PS)", "YPTF.MoE+Multi (noPS)", "YPTF.MoE+Multi (PS)"]
    precision = '32' # @param ["32", "bf16-mixed", "16"]
    project = '2024'

    if model_name == "YMT3+":
        checkpoint = "notask_all_cross_v6_xk2_amp0811_gm_ext_plus_nops_b72@model.ckpt"
        args = [checkpoint, '-p', project, '-pr', precision]
    elif model_name == "YPTF+Single (noPS)":
        checkpoint = "ptf_all_cross_rebal5_mirst_xk2_edr005_attend_c_full_plus_b100@model.ckpt"
        args = [checkpoint, '-p', project, '-enc', 'perceiver-tf', '-ac', 'spec',
                '-hop', '300', '-atc', '1', '-pr', precision]
    elif model_name == "YPTF+Multi (PS)":
        checkpoint = "mc13_256_all_cross_v6_xk5_amp0811_edr005_attend_c_full_plus_2psn_nl26_sb_b26r_800k@model.ckpt"
        args = [checkpoint, '-p', project, '-tk', 'mc13_full_plus_256',
                '-dec', 'multi-t5', '-nl', '26', '-enc', 'perceiver-tf',
                '-ac', 'spec', '-hop', '300', '-atc', '1', '-pr', precision]
    elif model_name == "YPTF.MoE+Multi (noPS)":
        checkpoint = "mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b36_nops@last.ckpt"
        args = [checkpoint, '-p', project, '-tk', 'mc13_full_plus_256', '-dec', 'multi-t5',
                '-nl', '26', '-enc', 'perceiver-tf', '-sqr', '1', '-ff', 'moe',
                '-wf', '4', '-nmoe', '8', '-kmoe', '2', '-act', 'silu', '-epe', 'rope',
                '-rp', '1', '-ac', 'spec', '-hop', '300', '-atc', '1', '-pr', precision]
    elif model_name == "YPTF.MoE+Multi (PS)":
        checkpoint = "mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b80_ps2@model.ckpt"
        args = [checkpoint, '-p', project, '-tk', 'mc13_full_plus_256', '-dec', 'multi-t5',
                '-nl', '26', '-enc', 'perceiver-tf', '-sqr', '1', '-ff', 'moe',
                '-wf', '4', '-nmoe', '8', '-kmoe', '2', '-act', 'silu', '-epe', 'rope',
                '-rp', '1', '-ac', 'spec', '-hop', '300', '-atc', '1', '-pr', precision]
    else:
        raise ValueError(model_name)

    # Extension
    model = load_model_checkpoint(args=args)

    #%% open audio and transcribe
    print(f"Current working directory: {os.getcwd()}") # is src folder
    audio_directory = '../../data/guitarset_yourmt3_16k/audio_mono-mic/'

    # Progress of calculation
    i = 0
    fileListLength = len(os.listdir(audio_directory))

    # Close all plot windows
    plt.close('all')

    for file in os.listdir(audio_directory):
        # nur über solo-audiodateien iterieren
        if "solo" in file:
            filename = os.fsdecode(file)

            # if file
            audio_filepath = os.path.join(audio_directory, filename)

            # calculate notefile with modified process_audio and transcribe
            # pred_notes = process_audio_notes(model, audio_filepath)

            # work with GT_notes for development
            GT_notes = extract_GT(audio_filepath)


            ## Estimate GuitarString
            estString = estGuitarString(audio_filepath, GT_notes)

            i += 1
            progress = round(100 * i/fileListLength, 2)
            print("Progress:", progress, "%")


            # local debug
            if(i == 3):
                break


def estGuitarString(audio_filepath, GT_notes):
    # Load audio
    sr, sig = wavfile.read(audio_filepath)
    sig = sig/np.max(abs(sig)) # normalize

    for note in GT_notes:
        onsetSample = int(note.onset * sr)  # Umrechnung in Integer
        offsetSample = int(note.offset * sr)

        noteFreq = noteToFreq(note.pitch) # Convert Midi-Note to frequency

        # TODO: zero-padding to catch very short notes
        # Extract note section from signal
        noteSig = sig[onsetSample:offsetSample]

        # Parameter für Buffering
        W = int(np.round(sr * 0.05))  # Fenstergröße = 50 ms
        H = int(np.round(W / 2))  # Schrittweite = 50% Überlappung

        # Pufferung durchführen
        buffered_signal = np.lib.stride_tricks.sliding_window_view(noteSig, window_shape=W)
        buffered_signal = buffered_signal[::H]  # Hopsize anwenden

        # Zero-Padding des letzten Frames (falls nötig)
        if buffered_signal.shape[1] < W:  # Falls das letzte Frame zu kurz ist
            padding_length = W - buffered_signal.shape[1]
            buffered_signal[-1] = np.pad(buffered_signal[-1], (0, padding_length), mode='constant')

        # Windowing mit Hanning-Fenster
        hanning_window = scipy.signal.windows.hann(W, sym=True)  # Hanning-Fenster
        buffered_windowed_signal = buffered_signal * hanning_window  # Fenster anwenden


        # Frequenzachse vorbereiten
        freqs = rfftfreq(W, d=1 / sr)

        # Ergebnisse speichern
        results = []
        interpolated_freqs = []  # Liste zum Speichern der interpolierten Frequenzen

        # Analyse jedes Frames, die ersten vier werden ubersprungen (100ms)
        for frame_idx, frame in enumerate(buffered_windowed_signal[5:], start=5):
            # FFT und Magnitudenspektrum
            NOTESIG = rfft(frame)
            magnitude_spectrum = np.abs(NOTESIG)
            magnitude_spectrum /= max(magnitude_spectrum)  # Normalisieren

            # Peaks finden
            peak_indices, _ = find_peaks(magnitude_spectrum, prominence=0.1)
            peak_freqs = freqs[peak_indices]
            peak_magnitudes = magnitude_spectrum[peak_indices]

            # Peak nächst zur erwarteten Frequenz
            if len(peak_indices) > 0:
                peak_idx_in_spectrum = peak_indices[np.argmin(np.abs(peak_freqs - noteFreq))]
                closest_peak_freq = freqs[peak_idx_in_spectrum]
                closest_peak_magnitude = magnitude_spectrum[peak_idx_in_spectrum]

                # Parabolische Interpolation
                _, delta, newMax = parainterp(peak_idx_in_spectrum, magnitude_spectrum)
                deltaf = freqs[1] - freqs[0]
                paraIntFreq = closest_peak_freq + delta * deltaf

                # Ergebnisse speichern
                results.append({
                    "frame_idx": frame_idx,
                    "closest_peak_freq": closest_peak_freq,
                    "closest_peak_magnitude": closest_peak_magnitude,
                    "paraIntFreq": paraIntFreq
                })

                # Interpolierte Frequenz speichern
                interpolated_freqs.append(paraIntFreq)

                # TODO: Frequenzbestimmung über Phasenableitung


                # TODO: Beta berechnen




                # Optional: Plotten
                plt.figure(figsize=(10, 6))
                plt.plot(freqs, magnitude_spectrum, label=f"Frame {frame_idx}")
                plt.scatter(closest_peak_freq, closest_peak_magnitude, color='green',
                            label=f"Closest Peak: {closest_peak_freq:.2f} Hz")
                plt.scatter(paraIntFreq, newMax, color='red', label=f"Interpolated Peak: {paraIntFreq:.2f} Hz")
                plt.title(f"Frequency Spectrum (Frame {frame_idx})")
                plt.xlabel("Frequency (Hz)")
                plt.ylabel("Magnitude")
                plt.xscale("log")
                plt.grid()
                plt.legend()
                plt.show()

                print(f"Frame {frame_idx}: Erwartete Frequenz {noteFreq} Hz")
                print(f"Nächster Peak: {closest_peak_freq:.2f} Hz")
                print(f"Interpolierte Frequenz: {paraIntFreq:.2f} Hz\n")
            else:
                print(f"Frame {frame_idx}: Keine Peaks gefunden.\n")

        # Mittelwert der interpolierten Frequenzen berechnen
        if interpolated_freqs:
            mean_interpolated_freq = np.mean(interpolated_freqs)
            print(f"Durchschnittliche interpolierte Frequenz: {mean_interpolated_freq:.2f} Hz")
        else:
            print("Keine interpolierten Frequenzen gefunden.")




def parainterp(idx, data):

    # Überprüfen, ob der Index für die Interpolation gültig ist
    if idx <= 0 or idx >= len(data) - 1:
        # Kein gültiger Bereich für Interpolation
        delta = 0.0
        newMax = data[idx]
        return idx, delta, newMax

    xy = data[idx - 1:idx + 2]

    # mittlerer Idx muss das maximum sein und indices: 0, 1, 2
    p = np.argmax(xy) # das maximum muss doch laut definition der findpeaks der mittlere Index sein ...

    # Berechnung der parabolischen Verschiebung
    delta = 0.5 * (xy[p - 1] - xy[p + 1]) / (xy[p - 1] - 2 * xy[p] + xy[p + 1])
    newMax = xy[p] - 0.25 * (xy[p - 1] - xy[p + 1]) * delta

    return p, delta, newMax

def noteToFreq(note):
    a = 440 #frequency of A (common value is 440Hz)
    return (a / 32) * (2 ** ((note - 9) / 12))


# %%
if __name__ == "__main__":
    main()