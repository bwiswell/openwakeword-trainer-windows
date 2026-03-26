class Logger:

    LEN = 60
    PHASE = 1
    SPACER = '=' * LEN


    ### CLASS METHODS ###
    @classmethod
    def log (cls, msg: str = ''):
        print(msg)

    @classmethod
    def start_phase (cls, name: str):
        title = f'Phase {Logger.PHASE}: {name}'
        remaining = Logger.LEN - len(title) - 2
        before = remaining // 2
        after = remaining - before
        print(f'{Logger.SPACER}')
        print(f'{'-' * before} {title} {'-' * after}')
        Logger.PHASE += 1