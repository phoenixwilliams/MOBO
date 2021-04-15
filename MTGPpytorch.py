import math
import torch
import gpytorch
from matplotlib import pyplot as plt


class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.RBFKernel()

        # We learn an IndexKernel for 2 tasks
        # (so we'll actually learn 2x2=4 tasks with correlations)
        self.task_covar_module = gpytorch.kernels.IndexKernel(num_tasks=2, rank=1)

    def forward(self,x,i):
        mean_x = self.mean_module(x)

        # Get input-input covariance
        covar_x = self.covar_module(x)
        # Get task-task covariance
        covar_i = self.task_covar_module(i)
        # Multiply the two together to get the covariance we want
        covar = covar_x.mul(covar_i)

        return gpytorch.distributions.MultivariateNormal(mean_x, covar)


if __name__ == "__main__":
    import os
    from GlobalOptimisation.NonSeparable import Functions as ns
    from GlobalOptimisation.Separable import Functions as s

    problem1 = s.Ackley(None, -20, 0.2, 20)
    problem2 = s.Rastrigin(None)

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    training_iterations = 1500
    dimension = 1
    pb = [-50, 50]
    train_size = 10
    test_size = 10

    final_diff1 = 0
    final_diff2 = 0

    for i in range(10):
        print(i)
        train_x1 = torch.rand(size=(train_size, dimension))
        train_x2 = torch.rand(size=(train_size, dimension))

        train_y1 = torch.tensor([problem1(xi*(pb[1]-pb[0]) + pb[0]) for xi in train_x1])
        train_y2 = torch.tensor([problem2(xi*(pb[1]-pb[0]) + pb[0]) for xi in train_x2])

        likelihood = gpytorch.likelihoods.GaussianLikelihood()

        train_i_task1 = torch.full_like(train_x1, dtype=torch.long, fill_value=0)
        train_i_task2 = torch.full_like(train_x2, dtype=torch.long, fill_value=1)

        full_train_x = torch.cat([train_x1, train_x2])
        full_train_i = torch.cat([train_i_task1, train_i_task2])
        full_train_y = torch.cat([train_y1, train_y2])

        # Here we have two iterms that we're passing in as train_inputs
        model = MultitaskGPModel((full_train_x, full_train_i), full_train_y, likelihood)

        # Find optimal model hyperparameters
        model.train()
        likelihood.train()

        # Use the adam optimizer
        optimizer = torch.optim.Adam([
            {'params': model.parameters()},  # Includes GaussianLikelihood parameters
        ], lr=0.1)

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        for i in range(training_iterations):
            optimizer.zero_grad()
            output = model(full_train_x, full_train_i)
            loss = -mll(output, full_train_y)
            loss.backward()
            #print('Iter %d/1500 - Loss: %.3f' % (i + 1, loss.item()))
            optimizer.step()

        model.eval()
        model.likelihood.eval()

        test_x1 = torch.rand(size=(test_size, dimension))
        test_x2 = torch.rand(size=(test_size, dimension))

        test_i1 = torch.full_like(test_x1, dtype=torch.long, fill_value=0)
        test_i2 = torch.full_like(test_x2, dtype=torch.long, fill_value=1)

        test_y1 = torch.tensor([problem1(xi*(pb[1]-pb[0]) + pb[0]) for xi in test_x1])
        test_y2 = torch.tensor([problem2(xi*(pb[1]-pb[0]) + pb[0]) for xi in test_x2])

    #print(test_y1.shape, test_i1.shape, train_y1.shape, train_x1.shape, test_x1.shape)

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred_y1 = likelihood(model(test_x1, test_i1))
            observed_pred_y2 = likelihood(model(test_x2, test_i2))

        avg_diff1 = sum(abs(observed_pred_y1.mean.detach().numpy() - test_y1.numpy())) / len(test_x1)
        avg_diff2 = sum(abs(observed_pred_y2.mean.detach().numpy() - test_y2.numpy())) / len(test_x2)

        final_diff1 += avg_diff1
        final_diff2 += avg_diff2

    print(final_diff1/10)
    print(final_diff2/10)


