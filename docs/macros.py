def define_env(env):
    """Macros Hook"""

    @env.macro
    def raider_version():
        import RAiDER
        return RAiDER.__version__
