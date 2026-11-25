"""Read action observations, excluding explicit JSONL footer metadata."""
import json


def action_id(value):
    if isinstance(value,str):
        if value=='RESET':return 0
        if value.startswith('ACTION'):value=value[6:]
    result=int(value)
    if not 0<=result<=7:raise ValueError('unknown action ID')
    return result


def observations(path, *, require_initial_reset=True):
    """Stream real observations; do not synthesize an unknown initial state."""
    first=True
    with path.open() as stream:
        for line_number,line in enumerate(stream,1):
            data=json.loads(line)['data']
            has_action='action_input' in data;has_frame='frame' in data
            if not has_action and not has_frame:
                continue  # recording footer, not an environment interaction
            if not has_action or not has_frame:
                raise ValueError(f'incomplete action observation at line {line_number}')
            if first and require_initial_reset and action_id(data['action_input']['id'])!=0:
                raise ValueError('recording must start with a RESET observation')
            first=False
            yield data
    if first:raise ValueError('recording has no action observations')


def prefix(path, steps):
    from itertools import islice
    return list(islice(observations(path),steps+1))
