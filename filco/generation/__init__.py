from filco.generation.backend import GenerationBackend


def get_backend(cfg, model_path: str) -> GenerationBackend:
    from filco.generation.sglang_backend import SGLangBackend
    return SGLangBackend(cfg, model_path)
