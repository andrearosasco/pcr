import torch
from torch.nn import BCEWithLogitsLoss
from torch.optim import SGD

try:
    from open3d.cuda.pybind.geometry import PointCloud
except ImportError:
    print("Open3d CUDA not found!")
    from open3d.cpu.pybind.geometry import PointCloud


# TODO NOTE
# In order to see the visualization correctly, one should stay in the coordinate frame, looking towards +z with +x
# facing towards -x and +y facing towards -y


class PointCloudReconstructor:
    def __init__(self, model, grid_res, device="cuda"):
        self.model = model
        self.grid_res = grid_res
        self.device = device

    def refine_point_cloud(self, complete_pc, fast_weights, n=10, show_loss=False):
        """
        Uses adversarial attack to refine the generated complete point cloud
        Args:
            show_loss: if True, the loss is printed at each iteration (useful to find correct value)
            complete_pc: torch.Tensor(N, 3)
            fast_weights: List ( List ( Tensor ) )
            n: Int, number of adversarial steps
        Returns:
            pc: PointCloud
        """
        complete_pc = complete_pc.unsqueeze(0)  # add batch dimension
        complete_pc.requires_grad = True

        loss_function = BCEWithLogitsLoss(reduction='mean')
        optim = SGD([complete_pc], lr=0.5, momentum=0.9)

        for step in range(n):
            results = self.model.sdf(complete_pc, fast_weights)

            gt = torch.ones_like(results[..., 0], dtype=torch.float32)
            gt[:, :] = 1
            loss_value = loss_function(results[..., 0], gt)

            self.model.zero_grad()
            optim.zero_grad()
            loss_value.backward(inputs=[complete_pc])
            optim.step()
            # if show_loss:
            #     print('Loss ', loss_value.item())
        return complete_pc

    def reconstruct_point_cloud(self, partial):
        """
        Given a partial point cloud, it reconstructs it
        Args:
            partial: np.array(N, 3)
        Returns:
            selected: Torch.Tensor(N, 3)
            fast_weights: List( List( Torch.Tensor ) )
        """
        # Inference
        partial = torch.FloatTensor(partial).unsqueeze(0).to(self.device)
        selected, fast_weights = self.model(partial, step=self.grid_res)

        return selected, fast_weights
