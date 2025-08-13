import random

import argparse
from queue import Empty

from omegaconf import OmegaConf
from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO, emit

from self_forcing import SelfForcingEndlessGenerator


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5001)
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument("--checkpoint_path", type=str, default='./checkpoints/self_forcing_dmd.pt')
    parser.add_argument("--config_path", type=str, default='./configs/self_forcing_dmd.yaml')
    parser.add_argument('--trt', action='store_true')
    args = parser.parse_args()
    return args


app = Flask(__name__)
app.config['SECRET_KEY'] = 'frontend_buffered_demo'
socketio = SocketIO(app, cors_allowed_origins="*")
config = None
generator = None


# Socket.IO event handlers
@socketio.on('connect')
def handle_connect():
    print('Client connected')
    emit('status', {'message': 'Connected to frontend-buffered demo server'})


@socketio.on('start_generation')
def handle_start_generation(data):
    global generator

    if generator is None:
        emit('error', {'message': 'Generator not initialized'})
        return

    if generator.generation_active:
        emit('error', {'message': 'Generation already in progress'})
        return

    prompt = data.get('prompt', '')
    seed = data.get('seed', -1)
    if seed == -1:
        seed = random.randint(0, 2 ** 32)

    if not prompt:
        emit('error', {'message': 'Prompt is required'})
        return

    # Start generation and frame processing
    socketio.start_background_task(generator.endless_generation_task, prompt, seed)

    emit('status', {'message': 'Generation started - frames will be sent immediately'})


@socketio.on('new_prompt')
def handle_new_prompt(data):
    global generator
    prompt = data.get('prompt', '')
    generator.set_prompt(prompt)
    emit('status', {'message': 'Prompt updated.'})


@socketio.on('stop_generation')
def handle_stop_generation():
    global generator
    generator.stop_generating()
    emit('status', {'message': 'Generation stopped'})


@socketio.on('purge_cache')
def handle_purge_cache(data=None):
    global generator
    purge_strength = 3  # Default to original behavior
    if data and 'purge_strength' in data:
        try:
            purge_strength = int(data['purge_strength'])
            if not 0 <= purge_strength <= 3:
                emit('error', {'message': 'purge_strength must be between 0 and 3'})
                return
        except (ValueError, TypeError):
            emit('error', {'message': 'Invalid purge_strength value'})
            return
    
    generator.purge_cache(purge_strength)
    emit('status', {'message': f'Cache purged with strength {purge_strength}, continuing with fresh context'})


def process_frame_queue():
    global generator
    while True:
        try:
            base64_frame = generator.image_queue.get(timeout=1.0)
            generator.image_queue.task_done()
            socketio.emit('frame_ready', {'data': base64_frame})
        except Empty:
            continue


# Web routes


@app.route('/')
def index():
    return render_template('demo.html')


if __name__ == '__main__':
    args = parse_args()
    config = OmegaConf.load(args.config_path)
    default_config = OmegaConf.load("configs/default_config.yaml")
    config = OmegaConf.merge(default_config, config)
    generator = SelfForcingEndlessGenerator(args.checkpoint_path, config)

    socketio.start_background_task(process_frame_queue)

    print(f"🚀 Starting demo on http://{args.host}:{args.port}")
    socketio.run(app, host=args.host, port=args.port, debug=False)
