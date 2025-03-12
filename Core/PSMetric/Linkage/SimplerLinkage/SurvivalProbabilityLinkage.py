from Core.PSMetric.Linkage.SimplerLinkage.LinkageMetric import LocalBivariateLinkageMetric


class SurvivalProbabilityLinkage(LocalBivariateLinkageMetric):

    def __init__(self):
        super().__init__()


    def get_probabilities_of_survival(self) -> dict[tuple[int, ...], float]:
        raise NotImplemented


class EmpiricalSurvivalProbabilityLinkage(SurvivalProbabilityLinkage):

    tournament_size: int
    samples: int

    def __init__(self, tournament_size: int, samples: int):
        self.tournament_size = tournament_size
        self.samples = samples
        super().__init__()

    def get_probabilities_of_survival(self) -> dict[tuple[int, ...], float]:
        pass