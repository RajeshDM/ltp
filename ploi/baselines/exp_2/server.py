import argparse
import torch
import zmq

from pathlib import Path
from datasets import collate_no_label as collate

def _parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=13579, help="TCP port for socket connection.")
    parser.add_argument('--model', required=True, type=Path, help="Model to use for predictions.")
    return parser.parse_args()


def _receive_predicates(socket):
    num_predicates = int(socket.recv().decode())
    predicates = []
    for _ in range(num_predicates):
        name = socket.recv().decode()
        arity = int(socket.recv().decode())
        predicates.append((name, arity))
    return predicates


def _receive_objects(socket):
    num_objects = int(socket.recv().decode())
    objects = []
    for _ in range(num_objects):
        name = socket.recv().decode()
        objects.append(name)
    return objects


def _receive_grounded_predicates(socket, predicates):
    num_facts = int(socket.recv().decode())
    facts = []
    for _ in range(num_facts):
        predicate = socket.recv().decode()
        arity = predicates[predicate]
        arguments = []
        for _ in range(arity):
            object = socket.recv().decode()
            arguments.append(object)
        facts.append((predicate, arguments))
    return facts


if __name__ == "__main__":
    args = _parse_arguments()

    if args.model.exists():
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        connection_string = 'tcp://*:{}'.format(args.port)
        socket.bind(connection_string)

        from architecture import MaxModelBase as Model

        print('Server started... ')
        foo = _receive_predicates(socket)
        pred_arities = dict(foo)
        predicates = list(pred_arities.keys())
        pred_ids = dict(zip(predicates, range(0, len(predicates))))
        print('Received predicates')
        device = torch.cuda.current_device() if torch.cuda.is_available() else None
        model = Model.load_from_checkpoint(checkpoint_path=str(args.model), strict=False).to(device)
        print('Loaded model')
        objects = _receive_objects(socket)
        obj_ids = dict(zip(objects, range(0, len(objects))))
        print('Received objects')
        facts = _receive_grounded_predicates(socket, pred_arities)
        print('Received facts')
        goals = _receive_grounded_predicates(socket, pred_arities)
        print('Received goals')

        def encode(grounded_predicates, offset):
            return [(pred_ids[predicate] + offset, [obj_ids[object] for object in arguments]) for predicate, arguments in grounded_predicates]

        goal_offset = len(predicates)
        facts_encoded = encode(facts, 0)
        goals_encoded = encode(goals, goal_offset)

        model.eval()
        with torch.no_grad():
            while True:
                request = socket.recv().decode()
                if request == 'QUIT': break
                batch_size = int(request)
                states = []
                for _ in range(batch_size):
                    state = _receive_grounded_predicates(socket, pred_arities)
                    state_encoded = facts_encoded + goals_encoded + encode(state, 0)
                    states.append(state_encoded)

                # State values, i.e. estimated distance to goal.
                values = model(collate(states, device))

                for index in range(batch_size):
                    rank = float(values[index])
                    flags = zmq.SNDMORE if (index + 1) < batch_size else 0
                    socket.send_string(str(rank), flags=flags)
    else:
        print('Model does not exist. Exiting.')
