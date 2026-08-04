from abc import ABC, abstractmethod


class GenerationBackend(ABC):

    @abstractmethod
    def generate_phase1(
        self,
        prompts: list,
        n_vote: int,
        greedy_sampling: dict,
        sample_sampling: dict,
    ) -> tuple:
        pass

    @abstractmethod
    def generate_greedy(self, prompts: list, max_new_tokens: int) -> list:
        pass

    @abstractmethod
    def sync_weights(self, model, ckpt_path: str) -> None:
        pass

    @abstractmethod
    def shutdown(self) -> None:
        pass

    def pause(self) -> None:
        pass

    def resume(self) -> None:
        pass
