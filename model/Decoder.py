import torch
from torch import enable_grad
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from configs import Config

class Decoder:
    def __init__(self, sdf):

        self.sdf = sdf

    def __call__(self, fast_weights):
        with torch.enable_grad():
            old = self.sdf.training
            self.sdf.eval()

            batch_size = fast_weights[0][0].shape[0]
            refined_pred = torch.tensor(torch.randn(batch_size, 400_000, 3).cpu().detach().numpy() * 0.1, device=Config.General.device,
                                        requires_grad=True)

            loss_function = BCEWithLogitsLoss(reduction='mean')
            optim = Adam([refined_pred], lr=0.1)

            c1, c2, c3, c4 = 1, 0, 0, 0 #1, 0, 0  1, 1e3, 0 # 0, 1e4, 5e2
            new_points = [[] for _ in range(batch_size)]
            # refined_pred.detach().clone()
            for step in range(20):
                results = self.sdf(refined_pred, fast_weights)

                for i in range(batch_size):
                    new_points[i] += [refined_pred[i].detach().clone()[(torch.sigmoid(results[i]).squeeze() >= 0.70), :]]

                gt = torch.ones_like(results[..., 0], dtype=torch.float32)
                gt[:, :] = 1
                loss1 = c1 * loss_function(results[..., 0], gt)

                loss_value = loss1

                self.sdf.zero_grad()
                optim.zero_grad()
                loss_value.backward(inputs=[refined_pred])
                optim.step()

        selected = [torch.cat(points).squeeze() for points in new_points]
        res = torch.zeros([batch_size, 400_000, 3], device=Config.General.device)
        for i, s in enumerate(selected):
            # torch.sum(torch.sum(torch.cat([s > 0.5, s < -0.5], dim=1), dim=1) != 0)
            k = min(s.size(0), 400_000)
            perm = torch.randperm(s.size(0))
            res[i][:k] = s[perm[:k]]

        self.sdf.train(old)
        return res

