import torch
from copy import deepcopy


class FedMoEClient(torch.nn.Module):
    def __init__(self, model, data_loader, device, lr=0.001, min_lr=1e-5, weight_decay=1e-5):
        super().__init__()
        self.model = model
        self.data_loader = data_loader
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=70, eta_min=min_lr
        )
        self.loss_func = torch.nn.BCELoss()
        self.device = device
        self.eta_l = lr

        self.SVR_layers = ["FC_3_shared", "FC_2_shared", "FC_1_shared", "FC_1", "FC_2", "FC_3"]
        self.control_variate = self._zero_like_model_params()
        self.param_mask = self._create_parameter_mask()

        self.control_variate = [c.to(self.device) for c in self.control_variate]
        self.param_mask = [p.to(self.device) for p in self.param_mask]

    def _zero_like_model_params(self):
        return [torch.zeros_like(p) for p in self.model.parameters()]

    def _create_parameter_mask(self):
        mask = []
        for name, param in self.model.named_parameters():
            is_svr = any(layer in name for layer in self.SVR_layers)
            mask.append(torch.ones_like(param) if is_svr else torch.zeros_like(param))
        return mask

    def train(self, global_controls, rounds=1, stage=1):
        global_controls = [c.to(self.device) for c in global_controls]
        initial_params = [p.detach().clone() for p in self.model.parameters()]

        self.model.train()
        total_avg_loss = 0.0
        for _ in range(rounds):
            running_loss = 0.0
            for inputs, labels in self.data_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                labels = torch.unsqueeze(labels, dim=1)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.loss_func(outputs, labels)
                loss.backward()

                with torch.no_grad():
                    for idx, param in enumerate(self.model.parameters()):
                        if self.param_mask[idx].sum() > 0:
                            param.grad -= (self.control_variate[idx] - global_controls[idx])

                self.optimizer.step()
                running_loss += loss.item()
            total_avg_loss += running_loss / len(self.data_loader)
        self.scheduler.step()

        with torch.no_grad():
            new_controls = []
            curr_params = list(self.model.parameters())
            for idx, (init_p, curr_p, c) in enumerate(
                zip(initial_params, curr_params, self.control_variate)
            ):
                if self.param_mask[idx].sum() > 0:
                    delta = (init_p - curr_p) / (rounds * self.eta_l)
                    new_c = c - global_controls[idx] + delta
                    new_controls.append(new_c)
                else:
                    new_controls.append(c)
        self.control_variate = new_controls
        return total_avg_loss / rounds

    def get_feature_extractor_weights(self):
        shared_weights = {
            "shared_experts": deepcopy(self.model.shared_experts.state_dict()),
            "FC_1_shared": deepcopy(self.model.FC_1_shared.state_dict()),
            "FC_2_shared": deepcopy(self.model.FC_2_shared.state_dict()),
            "FC_3_shared": deepcopy(self.model.FC_3_shared.state_dict()),
            "Drop_shared": deepcopy(self.model.Drop_shared.state_dict()),
        }
        return shared_weights, deepcopy(self.control_variate)

    def set_feature_extractor_weights(self, fe_weights):
        personalized_params = {
            "Conv_1": deepcopy(self.model.Conv_1.state_dict()),
            "Pool_1": deepcopy(self.model.Pool_1.state_dict()),
            "Conv_2": deepcopy(self.model.Conv_2.state_dict()),
            "Pool_2": deepcopy(self.model.Pool_2.state_dict()),
            "LSTM": deepcopy(self.model.LSTM.state_dict()),
            "FC_1": deepcopy(self.model.FC_1.state_dict()),
            "FC_2": deepcopy(self.model.FC_2.state_dict()),
            "FC_3": deepcopy(self.model.FC_3.state_dict()),
            "Drop": deepcopy(self.model.Drop.state_dict()),
            "gate": deepcopy(self.model.gate.state_dict()),
            "personalized_experts": deepcopy(self.model.personalized_experts.state_dict()),
            "Gating": deepcopy(self.model.Gating.state_dict()),
        }

        self.model.shared_experts.load_state_dict(fe_weights["shared_experts"])
        self.model.FC_1_shared.load_state_dict(fe_weights["FC_1_shared"])
        self.model.FC_2_shared.load_state_dict(fe_weights["FC_2_shared"])
        self.model.FC_3_shared.load_state_dict(fe_weights["FC_3_shared"])
        self.model.Drop_shared.load_state_dict(fe_weights["Drop_shared"])

        self.model.Conv_1.load_state_dict(personalized_params["Conv_1"])
        self.model.Pool_1.load_state_dict(personalized_params["Pool_1"])
        self.model.Conv_2.load_state_dict(personalized_params["Conv_2"])
        self.model.Pool_2.load_state_dict(personalized_params["Pool_2"])
        self.model.LSTM.load_state_dict(personalized_params["LSTM"])
        self.model.FC_1.load_state_dict(personalized_params["FC_1"])
        self.model.FC_2.load_state_dict(personalized_params["FC_2"])
        self.model.FC_3.load_state_dict(personalized_params["FC_3"])
        self.model.Drop.load_state_dict(personalized_params["Drop"])
        self.model.gate.load_state_dict(personalized_params["gate"])
        self.model.personalized_experts.load_state_dict(personalized_params["personalized_experts"])
        self.model.Gating.load_state_dict(personalized_params["Gating"])


class FedMoEServer(torch.nn.Module):
    def __init__(self, global_model, agg_weights, device, eta_g=1.0):
        super().__init__()
        self.global_model = global_model
        weights_tensor = torch.tensor(agg_weights, dtype=torch.float).to(device)
        self.agg_weights = weights_tensor / weights_tensor.sum()
        self.device = device
        self.eta_g = eta_g
        self.control_variate = self._zero_like_model_params()

    def _zero_like_model_params(self):
        return [torch.zeros_like(p) for p in self.global_model.parameters()]

    def aggregate_fe(self, client_fe_weights_list, client_losses=None, client_controls=None):
        if client_losses is not None:
            assert len(client_fe_weights_list) == len(client_losses)
            loss_tensor = torch.tensor(client_losses, dtype=torch.float)
            normalized_loss = loss_tensor / loss_tensor.sum()
            inverted_loss = 1.0 - normalized_loss
            new_agg_weights = inverted_loss / inverted_loss.sum()
            self.agg_weights = new_agg_weights.to(self.device)

        assert len(client_fe_weights_list) == len(self.agg_weights)

        global_fe_weights = {
            "shared_experts": deepcopy(self.global_model.shared_experts.state_dict()),
            "FC_1_shared": deepcopy(self.global_model.FC_1_shared.state_dict()),
            "FC_2_shared": deepcopy(self.global_model.FC_2_shared.state_dict()),
            "FC_3_shared": deepcopy(self.global_model.FC_3_shared.state_dict()),
            "Drop_shared": deepcopy(self.global_model.Drop_shared.state_dict()),
        }

        for layer in ["shared_experts", "FC_1_shared", "FC_2_shared", "FC_3_shared", "Drop_shared"]:
            for key in global_fe_weights[layer]:
                stacked_weights = torch.stack(
                    [client_fe_weights[layer][key].float() for client_fe_weights in client_fe_weights_list], 0
                )
                agg_w = self.agg_weights.view(-1, *([1] * (stacked_weights.dim() - 1)))
                global_fe_weights[layer][key] = (stacked_weights * agg_w).sum(dim=0)

        self.global_model.shared_experts.load_state_dict(global_fe_weights["shared_experts"])

        new_controls = []
        for i in range(len(self.control_variate)):
            client_c = [ctrl[i] for ctrl in client_controls]
            new_controls.append(torch.stack(client_c).mean(dim=0))
        self.control_variate = new_controls

    def get_global_feature_extractor_weights(self):
        shared_weights = {
            "shared_experts": deepcopy(self.global_model.shared_experts.state_dict()),
            "FC_1_shared": deepcopy(self.global_model.FC_1_shared.state_dict()),
            "FC_2_shared": deepcopy(self.global_model.FC_2_shared.state_dict()),
            "FC_3_shared": deepcopy(self.global_model.FC_3_shared.state_dict()),
            "Drop_shared": deepcopy(self.global_model.Drop_shared.state_dict()),
        }
        return shared_weights, deepcopy(self.control_variate)
