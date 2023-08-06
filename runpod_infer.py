import os
import predict

import runpod
from runpod.serverless.utils import rp_download, rp_upload, rp_cleanup
from runpod.serverless.utils.rp_validator import validate


MODEL = predict.Predictor()
MODEL.setup()


INPUT_SCHEMA = {
    'model_version': {
        'type': str,
        'required': False,
        'default': 'melody',
        'constraints': lambda model_version: model_version in ["melody", "large", "encode-decode"]
    },
    'prompt': {
        'type': str,
        'required': False,
        'default': None
    },
    'input_audio': {
        'type': str,
        'required': False,
        'default': None
    },
    'duration': {
        'type': int,
        'required': False,
        'default': 8,
        'constraints': lambda duration: duration>8 and duration<30
    },
    'continuation': {
        'type': bool,
        'required': False,
        'default': False
    },
    'continuation_start': {
        'type': int,
        'required': False,
        'default': 0,
        'constraints': lambda continuation_start: continuation_start>=0
    },
    'continuation_end': {
        'type': int,
        'required': False,
        'default': None,
        'constraints': lambda continuation_end: continuation_end>=0
    },
    'normalization_strategy': {
        'type': str,
        'required': False,
        'default': 'loudness',
        'constraints': lambda normalization_strategy: normalization_strategy in ["loudness", "clip", "peak", "rms"]
    },
    'top_k': {
        'type': int,
        'required': False,
        'default': 250
    },
    'top_p': {
        'type': float,
        'required': False,
        'default': 0.0
    },
    'temperature': {
        'type': float,
        'required': False,
        'default': 1.0
    },
    'classifier_free_guidance': {
        'type': int,
        'required': False,
        'default': 3
    },
    'output_format': {
        'type': str,
        'required': False,
        'default': 'wav',
        'constraints': lambda output_format: output_format in ["wav", "mp3"]
    },
    'seed': {
        'type': int,
        'required': False,
        'default': None
    }
}

def run(job):
    '''
    Run inference on the model.
    Returns output path, width the seed used to generate the image.
    '''
    job_input = job['input']

    # Input validation
    validated_input = validate(job_input, INPUT_SCHEMA)

    if 'errors' in validated_input:
        return {"error": validated_input['errors']}
    validated_input = validated_input['validated_input']

    job_input['input_audio'] = rp_download.download_files_from_urls(
        job['id'],
        [job_input.get('input_audio', None)]
    )  # pylint: disable=unbalanced-tuple-unpacking

    audio_path = MODEL.predict(
        model_version=validated_input['model_version'],
        prompt=validated_input['prompt'],
        input_audio=validated_input['input_audio'],
        duration=validated_input['duration'],
        continuation=validated_input['continuation'],
        continuation_start=validated_input['continuation_start'],
        continuation_end=validated_input['continuation_end'],
        normalization_strategy=validated_input['normalization_strategy'],
        top_k=validated_input['top_k'],
        top_p=validated_input['top_p'],
        temperature=validated_input['temperature'],
        classifier_free_guidance=validated_input['classifier_free_guidance'],
        output_format=validated_input['output_format'],
        seed=validated_input['seed']
    )

    job_output = []

    audio_url = rp_upload.upload_image(job['id'], audio_path, 0)

    job_output.append({
        "audio": audio_url,
    })
    
    # Remove downloaded input objects
    rp_cleanup.clean(['input_objects'])

    return job_output

runpod.serverless.start({"handler": run})