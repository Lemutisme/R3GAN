import torch
import torch.nn as nn

class AdversarialTraining:
    def __init__(self, Generator, Discriminator):
        self.Generator = Generator
        self.Discriminator = Discriminator
        
    @staticmethod
    def ZeroCenteredGradientPenalty(Samples, Critics):
        Gradient, = torch.autograd.grad(outputs=Critics.sum(), inputs=Samples, create_graph=True)
        return Gradient.square().sum([1, 2, 3])
        
    def AccumulateGeneratorGradients(self, Noise, RealSamples, Conditions, Scale=1, Preprocessor=lambda x: x):
        FakeSamples = self.Generator(Noise, Conditions)
        RealSamples = RealSamples.detach()
        
        FakeLogits = self.Discriminator(Preprocessor(FakeSamples), Conditions)
        RealLogits = self.Discriminator(Preprocessor(RealSamples), Conditions)
        
        RelativisticLogits = FakeLogits - RealLogits
        AdversarialLoss = nn.functional.softplus(-RelativisticLogits)
        
        (Scale * AdversarialLoss.mean()).backward()
        
        return [x.detach() for x in [AdversarialLoss, RelativisticLogits]]
    
    def AccumulateDiscriminatorGradients(
        self, Noise, RealSamples, Conditions, Gamma, Scale=1, Preprocessor=lambda x: x,
        AdversarialScale=1.0, UseR1Penalty=True, UseR2Penalty=True
    ):
        RealSamples = RealSamples.detach().requires_grad_(UseR1Penalty)
        FakeSamples = self.Generator(Noise, Conditions).detach().requires_grad_(UseR2Penalty)
        
        RealLogits = self.Discriminator(Preprocessor(RealSamples), Conditions)
        FakeLogits = self.Discriminator(Preprocessor(FakeSamples), Conditions)
        
        R1Penalty = torch.zeros([RealLogits.shape[0]], device=RealLogits.device, dtype=RealLogits.dtype)
        R2Penalty = torch.zeros([FakeLogits.shape[0]], device=FakeLogits.device, dtype=FakeLogits.dtype)
        if UseR1Penalty:
            R1Penalty = AdversarialTraining.ZeroCenteredGradientPenalty(RealSamples, RealLogits)
        if UseR2Penalty:
            R2Penalty = AdversarialTraining.ZeroCenteredGradientPenalty(FakeSamples, FakeLogits)
        
        RelativisticLogits = RealLogits - FakeLogits
        AdversarialLoss = nn.functional.softplus(-RelativisticLogits)
        R1Penalty = R1Penalty.reshape_as(AdversarialLoss)
        R2Penalty = R2Penalty.reshape_as(AdversarialLoss)
        
        DiscriminatorLoss = AdversarialScale * AdversarialLoss + (Gamma / 2) * (R1Penalty + R2Penalty)
        (Scale * DiscriminatorLoss.mean()).backward()
        
        return [x.detach() for x in [AdversarialLoss, RelativisticLogits, R1Penalty, R2Penalty]]
