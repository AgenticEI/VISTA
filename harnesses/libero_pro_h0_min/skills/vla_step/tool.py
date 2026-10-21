from rpent.harness import ToolBinding


def build(runtime):
    def vla_step(instruction: str):
        return runtime.call('chunk', {"instruction": instruction})

    return ToolBinding(
        'vla_step',
        'Execute exactly one raw Pi0.5 action batch for instruction and return the resulting observation. No grasp loop or recovery strategy.',
        {'type': 'object', 'properties': {'instruction': {'type': 'string', 'minLength': 1, 'maxLength': 4000}}, 'required': ['instruction'], 'additionalProperties': False},
        vla_step,
    )
