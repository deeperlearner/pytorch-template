import torch


class Tester:
    """
    Tester class
    """

    def __init__(self, test_data_loaders, models, device, metrics_epoch, test_metrics):
        self.test_data_loaders = test_data_loaders
        self.model = models["model"]
        self.device = device
        self.metrics_epoch = metrics_epoch
        self.test_metrics = test_metrics

    def test(self):
        self.model.eval()
        with torch.no_grad():
            print("testing...")
            test_loader = self.test_data_loaders["data"]

            if len(self.metrics_epoch) > 0:
                outputs = torch.FloatTensor().to(self.device)
                targets = torch.FloatTensor().to(self.device)
            for batch_idx, (data, target) in enumerate(test_loader):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                if len(self.metrics_epoch) > 0:
                    outputs = torch.cat((outputs, output))
                    targets = torch.cat((targets, target))

                #
                # save sample images, or do something with output here
                #

            for met in self.metrics_epoch:
                self.test_metrics.epoch_update(met.__name__, met(targets, outputs))

        return targets, outputs, self.test_metrics.result()
