from elevenlabs.client import ElevenLabs
from utils import setup_logging

logger = setup_logging()


def generate_audio(questions, api_key):
    client = ElevenLabs(api_key=api_key)

    for i, question in enumerate(questions, 1):
        audio_stream = client.generate(
            text=question,
            model="eleven_turbo_v2",
            stream=True
        )

        with open(f"src/content/sample_data/audio/tutorial{i}.mp3", "wb") as f:
            for chunk in audio_stream:
                f.write(chunk)

        logger.info(f"Audio file tutorial{i}.mp3 created")