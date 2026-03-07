import torch
import torch.nn as nn


class AdversarialTraining:
    def __init__(self, Generator, Discriminator):
        self.Generator = Generator
        self.Discriminator = Discriminator

    @staticmethod
    def _as_vector(x):
        if x.ndim == 1:
            return x
        return x.reshape(x.shape[0], -1).mean(dim=1)

    @staticmethod
    def ZeroCenteredGradientPenalty(Samples, Critics):
        (Gradient,) = torch.autograd.grad(
            outputs=Critics.sum(), inputs=Samples, create_graph=True
        )
        return Gradient.square().sum([1, 2, 3])

    @staticmethod
    def _pairwise_delta(RealLogits, FakeLogits):
        return RealLogits - FakeLogits

    @staticmethod
    def _generator_adv_loss(
        FakeLogits, RealLogits, LossType="softmargin", Margin=0.0, Tau=0.07
    ):
        Delta = AdversarialTraining._pairwise_delta(RealLogits, FakeLogits)
        if LossType == "softmargin":
            RelativisticLogits = -Delta
            AdversarialLoss = nn.functional.softplus(Margin + Delta)
            return AdversarialLoss, RelativisticLogits

        if LossType == "infonce":
            if Tau <= 0:
                raise ValueError(f"InfoNCE temperature must be positive, got {Tau}")
            fake_scaled = FakeLogits / Tau
            real_scaled = RealLogits / Tau
            logsum_real = torch.logsumexp(real_scaled, dim=0)
            AdversarialLoss = nn.functional.softplus(logsum_real - fake_scaled)
            RelativisticLogits = FakeLogits - RealLogits
            return AdversarialLoss, RelativisticLogits

        raise ValueError(f"Unknown adversarial loss type: {LossType}")

    @staticmethod
    def _discriminator_adv_loss(
        RealLogits, FakeLogits, LossType="softmargin", Margin=0.0, Tau=0.07
    ):
        Delta = AdversarialTraining._pairwise_delta(RealLogits, FakeLogits)
        if LossType == "softmargin":
            RelativisticLogits = Delta
            AdversarialLoss = nn.functional.softplus(Margin - Delta)
            return AdversarialLoss, RelativisticLogits

        if LossType == "infonce":
            if Tau <= 0:
                raise ValueError(f"InfoNCE temperature must be positive, got {Tau}")
            real_scaled = RealLogits / Tau
            fake_scaled = FakeLogits / Tau
            logsum_fake = torch.logsumexp(fake_scaled, dim=0)
            AdversarialLoss = nn.functional.softplus(logsum_fake - real_scaled)
            RelativisticLogits = RealLogits - FakeLogits
            return AdversarialLoss, RelativisticLogits

        raise ValueError(f"Unknown adversarial loss type: {LossType}")

    def AccumulateGeneratorGradients(
        self,
        Noise,
        RealSamples,
        Conditions,
        Scale=1,
        Preprocessor=lambda x: x,
        Margin=0.0,
        LossType="softmargin",
        Tau=0.07,
        AdversarialScale=1.0,
    ):
        FakeSamples = self.Generator(Noise, Conditions)
        RealSamples = RealSamples.detach()

        FakeLogits = self._as_vector(
            self.Discriminator(Preprocessor(FakeSamples), Conditions)
        )
        RealLogits = self._as_vector(
            self.Discriminator(Preprocessor(RealSamples), Conditions)
        )

        AdversarialLoss, RelativisticLogits = self._generator_adv_loss(
            FakeLogits=FakeLogits,
            RealLogits=RealLogits,
            LossType=LossType,
            Margin=Margin,
            Tau=Tau,
        )

        GeneratorLoss = AdversarialScale * AdversarialLoss
        (Scale * GeneratorLoss.mean()).backward()

        return [x.detach() for x in [AdversarialLoss, RelativisticLogits]]

    def AccumulateDiscriminatorGradients(
        self,
        Noise,
        RealSamples,
        Conditions,
        Gamma,
        Scale=1,
        Preprocessor=lambda x: x,
        AdversarialScale=1.0,
        UseR1Penalty=True,
        UseR2Penalty=True,
        UseNonAugGP=False,
        Margin=0.0,
        LossType="softmargin",
        Tau=0.07,
    ):
        RealSamples = RealSamples.detach().requires_grad_(UseR1Penalty)
        FakeSamples = (
            self.Generator(Noise, Conditions).detach().requires_grad_(UseR2Penalty)
        )

        RealLogits = self._as_vector(
            self.Discriminator(Preprocessor(RealSamples), Conditions)
        )
        FakeLogits = self._as_vector(
            self.Discriminator(Preprocessor(FakeSamples), Conditions)
        )

        R1Penalty = torch.zeros(
            [RealLogits.shape[0]], device=RealLogits.device, dtype=RealLogits.dtype
        )
        R2Penalty = torch.zeros(
            [FakeLogits.shape[0]], device=FakeLogits.device, dtype=FakeLogits.dtype
        )
        if UseR1Penalty:
            RealLogitsForPenalty = RealLogits
            if UseNonAugGP:
                RealLogitsForPenalty = self._as_vector(
                    self.Discriminator(RealSamples, Conditions)
                )
            R1Penalty = AdversarialTraining.ZeroCenteredGradientPenalty(
                RealSamples, RealLogitsForPenalty
            )
        if UseR2Penalty:
            FakeLogitsForPenalty = FakeLogits
            if UseNonAugGP:
                FakeLogitsForPenalty = self._as_vector(
                    self.Discriminator(FakeSamples, Conditions)
                )
            R2Penalty = AdversarialTraining.ZeroCenteredGradientPenalty(
                FakeSamples, FakeLogitsForPenalty
            )

        AdversarialLoss, RelativisticLogits = self._discriminator_adv_loss(
            RealLogits=RealLogits,
            FakeLogits=FakeLogits,
            LossType=LossType,
            Margin=Margin,
            Tau=Tau,
        )
        R1Penalty = R1Penalty.reshape_as(AdversarialLoss)
        R2Penalty = R2Penalty.reshape_as(AdversarialLoss)

        DiscriminatorLoss = AdversarialScale * AdversarialLoss + (Gamma / 2) * (
            R1Penalty + R2Penalty
        )
        (Scale * DiscriminatorLoss.mean()).backward()

        return [
            x.detach()
            for x in [
                AdversarialLoss,
                RelativisticLogits,
                R1Penalty,
                R2Penalty,
                FakeSamples,
            ]
        ]
