from rpent.harness import ToolBinding


def build(runtime):
    def observe():
        return runtime.call('sensor', {})

    return ToolBinding(
        'observe',
        'Read current task, public robot state and both camera images without acting.',
        {'type': 'object', 'properties': {}, 'additionalProperties': False},
        observe,
    )
