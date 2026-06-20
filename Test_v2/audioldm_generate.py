import argparse
import soundfile as sf

from diffusers import AudioLDM2Pipeline

parser = argparse.ArgumentParser()

parser.add_argument("--prompt", required=True)
parser.add_argument("--length", type=float, required=True)
parser.add_argument("--output", required=True)

args = parser.parse_args()

pipe = AudioLDM2Pipeline.from_pretrained("cvssp/audioldm2")

audio = pipe(
    prompt=args.prompt,
    audio_length_in_s=args.length,
    num_inference_steps=50
).audios[0]

sf.write(args.output, audio, 16000)
