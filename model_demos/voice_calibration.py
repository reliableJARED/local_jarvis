"""
Voice Calibration Example - READY TO RUN
-----------------------------------------
This script calibrates a voice recognition system using TTS-generated audio.

Requirements:
    - BrocasArea class from your first file
    - VoiceRecognitionSystem class from your second file
    - scipy (for audio resampling)

Process:
1. Generate 10 diverse phrases using BrocasArea TTS (without playing)
2. Convert audio from 24kHz to 16kHz for voice recognition
3. Create a new voice profile with id "my_voice_id"
"""

import numpy as np
from scipy import signal
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Import the systems
from brocasArea import BrocasArea
from voicerecognition import VoiceRecognitionSystem


def resample_audio(audio_data: np.ndarray, orig_rate: int, target_rate: int) -> np.ndarray:
    """
    Resample audio from one sample rate to another.
    
    Args:
        audio_data: Input audio data
        orig_rate: Original sample rate (e.g., 24000)
        target_rate: Target sample rate (e.g., 16000)
        
    Returns:
        Resampled audio data
    """
    # Calculate the number of samples in the resampled audio
    num_samples = int(len(audio_data) * target_rate / orig_rate)
    
    # Use scipy's resample function for high-quality resampling
    resampled = signal.resample(audio_data, num_samples)
    
    return resampled.astype(np.float32)


def run_calibration():
    """
    Run the voice calibration process.
    """
    logging.info("Starting voice calibration process...")
    
    # 10 diverse phrases covering different phonetic patterns
    calibration_phrases = [
        "The quick brown fox jumps over the lazy dog.",
        "How much wood would a woodchuck chuck if a woodchuck could chuck wood?",
        "She sells seashells by the seashore.",
        "Peter Piper picked a peck of pickled peppers.",
        "I scream, you scream, we all scream for ice cream!",
        "Unique New York, you need New York, you know you need unique New York.",
        "Red leather, yellow leather, red leather, yellow leather.",
        "The sixth sick sheikh's sixth sheep's sick.",
        "Can you can a can as a canner can can a can?",
        "How can a clam cram in a clean cream can?"
    ]
    
    logging.info(f"Prepared {len(calibration_phrases)} calibration phrases")
    
    # Initialize BrocasArea TTS system
    logging.info("Initializing BrocasArea TTS system...")
    tts = BrocasArea()
    
    # Initialize Voice Recognition System
    logging.info("Initializing Voice Recognition System...")
    vrs = VoiceRecognitionSystem(
        db_path="voice_profiles.db",
        use_gpu=False,
        max_samples_per_profile=10
    )
    
    # Profile ID for the calibration
    profile_id = "my_voice_id"
    
    # Process each phrase
    audio_samples = []
    logging.info("\n" + "="*50)
    logging.info("SYNTHESIZING CALIBRATION PHRASES")
    logging.info("="*50)
    
    for i, phrase in enumerate(calibration_phrases, 1):
        logging.info(f"\nPhrase {i}/{len(calibration_phrases)}: {phrase[:50]}...")
        
        # Synthesize speech (DO NOT PLAY - auto_play=False)
        logging.info("  → Synthesizing speech via BrocasArea...")
        audio_dict = tts.synthesize_speech(phrase, auto_play=False)
        
        if audio_dict is None:
            logging.error(f"  ✗ Failed to synthesize phrase {i}")
            continue
        
        # Extract audio data
        audio_data_24k = audio_dict['audio_data']
        orig_rate = audio_dict['samplerate']  # Should be 24000 Hz
        
        # Convert from 24kHz to 16kHz for voice recognition
        logging.info(f"  → Converting from {orig_rate}Hz to 16kHz...")
        audio_data_16k = resample_audio(audio_data_24k, orig_rate, 16000)
        
        # Store the resampled audio
        audio_samples.append(audio_data_16k)
        
        logging.info(f"  ✓ Phrase {i} processed successfully")
        logging.info(f"    Original length: {len(audio_data_24k)} samples @ {orig_rate}Hz")
        logging.info(f"    Resampled length: {len(audio_data_16k)} samples @ 16000Hz")
    
    # Create voice profile with all samples
    logging.info("\n" + "="*50)
    logging.info("CREATING VOICE PROFILE")
    logging.info("="*50)
    logging.info(f"\nProfile ID: {profile_id}")
    logging.info(f"Number of samples: {len(audio_samples)}")
    
    success = vrs.add_speaker_profile(
        human_id=profile_id,
        audio_samples=audio_samples,
        sample_rate=16000
    )
    
    if success:
        logging.info(f"\n✓ Voice profile '{profile_id}' created successfully!")
        
        # Get profile information
        profile_info = vrs.get_speaker_profile(profile_id)
        quality_metrics = vrs.get_voice_sample_quality_metrics(profile_id)
        
        logging.info(f"\nProfile Information:")
        logging.info(f"  - ID: {profile_info['human_id']}")
        logging.info(f"  - Samples: {profile_info['samples_count']}")
        logging.info(f"  - Created: {profile_info['created_at']}")
        
        logging.info(f"\nQuality Metrics:")
        logging.info(f"  - Cluster Quality: {quality_metrics['cluster_quality']}")
        logging.info(f"  - Avg Cohesion Score: {quality_metrics['avg_cohesion_score']:.3f}")
        logging.info(f"  - Avg Distance to Centroid: {quality_metrics['avg_distance_to_centroid']:.3f}")
        
    else:
        logging.error(f"\n✗ Failed to create voice profile '{profile_id}'")
    
    # Cleanup
    logging.info("\n" + "="*50)
    logging.info("CLEANUP")
    logging.info("="*50)
    tts.shutdown()
    vrs.close()
    logging.info("✓ Systems shut down successfully")
    
    logging.info("\n" + "="*50)
    logging.info("CALIBRATION COMPLETE")
    logging.info("="*50)
    
    return success


if __name__ == "__main__":
    print("\n" + "="*70)
    print("VOICE CALIBRATION SYSTEM")
    print("="*70)
    print("\nThis script will:")
    print("  1. Synthesize 10 diverse phrases using BrocasArea TTS")
    print("  2. Convert audio from 24kHz to 16kHz")
    print("  3. Create a voice profile 'my_voice_id'")
    print("  4. Display quality metrics")
    print("\nNOTE: Audio will NOT be played during this process.")
    print("="*70 + "\n")
    
    try:
        success = run_calibration()
        
        if success:
            print("\n" + "🎉 Calibration completed successfully!")
            print("Voice profile 'my_voice_id' is ready to use.")
        else:
            print("\n" + "❌ Calibration failed. Check the logs above.")
            
    except Exception as e:
        logging.error(f"Fatal error during calibration: {e}", exc_info=True)
        print("\n" + "❌ Calibration failed with error. Check the logs above.")