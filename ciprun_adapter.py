class CIPRUNAdapter:

    def __init__(self):
        super().__init__()
        pass

    # def on_sanity_check(self, ctx, command):
    #     command.append('e.debug=true')
    #     return command

    def get_output_ignores(self):
        return [
            'wandb_logs',
            'wandb',
        ]

    # def on_launch(self, ctx, command):
    #     name = ctx.job_id
    #     command.append(f'e.name={name}')
    #     return command
